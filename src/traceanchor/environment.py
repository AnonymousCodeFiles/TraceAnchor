from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from traceanchor.config import TraceAnchorConfig, config_hash


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    expected: str
    actual: str
    required: bool = True


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode, output


def _cudnn_version(raw: int | None) -> str:
    if not raw:
        return "unavailable"
    major = raw // 1000
    minor = (raw % 1000) // 100
    return f"{major}.{minor}"


def collect_checks(config: TraceAnchorConfig, config_path: Path) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(Check("python", python_version == config.runtime.python, config.runtime.python, python_version))

    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_version = torch.__version__.split("+")[0]
        cuda_version = torch.version.cuda or "unavailable"
        cudnn_version = _cudnn_version(torch.backends.cudnn.version())
        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": cuda_version,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
        }
        checks.extend(
            [
                Check("pytorch", torch_version == config.runtime.pytorch, config.runtime.pytorch, torch_version),
                Check("cuda_runtime", cuda_version == config.runtime.cuda, config.runtime.cuda, cuda_version),
                Check("cudnn", cudnn_version == config.runtime.cudnn, config.runtime.cudnn, cudnn_version),
                Check("cuda_available", cuda_available, "true", str(cuda_available).lower()),
            ]
        )
    except (ImportError, OSError) as exc:
        checks.extend(
            [
                Check("pytorch", False, config.runtime.pytorch, f"unavailable: {exc}"),
                Check("cuda_runtime", False, config.runtime.cuda, "unavailable"),
                Check("cudnn", False, config.runtime.cudnn, "unavailable"),
                Check("cuda_available", False, "true", "false"),
            ]
        )

    resolved = config.resolved_dict(config_path)
    raw_root = Path(resolved["paths"]["raw_data_root"])
    example = Path(resolved["paths"]["example_scenario"])
    checks.extend(
        [
            Check("raw_data_root", raw_root.is_dir(), "existing readable directory", str(raw_root)),
            Check("example_scenario", example.is_dir(), "existing readable directory", str(example)),
        ]
    )

    disk = shutil.disk_usage(config.project.project_root)
    required_bytes = config.resources.disk_free_required_gib * 1024**3
    checks.append(
        Check(
            "disk_free_gib",
            disk.free >= required_bytes,
            f">={required_bytes / 1024**3:.0f}",
            f"{disk.free / 1024**3:.1f}",
        )
    )

    nvidia_code, nvidia_output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    git_code, git_output = _run(["git", "status", "--short", "--branch"])
    metadata = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_info,
        "nvidia_smi": {"returncode": nvidia_code, "output": nvidia_output},
        "git": {"returncode": git_code, "status": git_output},
        "disk_free_bytes": disk.free,
    }
    return checks, metadata


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_environment_manifest(
    config: TraceAnchorConfig,
    config_path: Path,
    checks: list[Check],
    metadata: dict[str, Any],
) -> Path:
    resolved = config.resolved_dict(config_path)
    manifests_dir = Path(resolved["paths"]["manifests_dir"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash(config, config_path),
        "hard_requirements_met": all(check.ok for check in checks if check.required),
        "checks": [asdict(check) for check in checks],
        "environment": metadata,
    }
    _atomic_write_text(
        manifests_dir / "environment.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    freeze_code, freeze_output = _run([sys.executable, "-m", "pip", "freeze"])
    if freeze_code != 0:
        freeze_output = f"pip freeze failed ({freeze_code}): {freeze_output}"
    _atomic_write_text(manifests_dir / "pip_freeze.txt", freeze_output + "\n")
    _atomic_write_text(
        manifests_dir / "gpu_info.txt",
        metadata["nvidia_smi"]["output"] + "\n",
    )
    _atomic_write_text(
        manifests_dir / "config_resolved.yml",
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=False),
    )
    return manifests_dir / "environment.json"
