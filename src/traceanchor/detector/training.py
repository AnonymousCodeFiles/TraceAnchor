from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.detector.common import focal_bce_with_logits
from traceanchor.detector.data import (
    CausalSceneDataset,
    FeatureSpec,
    class_counts,
    load_feature_spec,
    load_scenes,
)
from traceanchor.detector.host_tcn import HostTCN
from traceanchor.detector.metrics import scene_macro_metrics
from traceanchor.detector.network_tcn import NetworkTCN
from traceanchor.ingest.common import atomic_write_json, sha256_file


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_model(
    config: TraceAnchorConfig, channel: str, input_dim: int
) -> nn.Module:
    common = config.detector.common
    arguments = {
        "input_dim": input_dim,
        "channels": common.channels,
        "kernel_size": common.kernel_size,
        "dilations": tuple(common.dilations),
        "dropout": common.dropout,
    }
    if channel == "network":
        return NetworkTCN(**arguments)
    if channel == "host":
        return HostTCN(
            **arguments,
            hidden_dim=config.detector.host.pre_projection_hidden_dim,
        )
    raise ValueError(f"invalid detector channel: {channel}")


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-repository"


def _feature_manifest_hash(features_dir: Path) -> str:
    paths = [
        features_dir / "index" / "train.json",
        features_dir / "index" / "validation.json",
        features_dir / "scalers" / "network_scaler.json",
        features_dir / "scalers" / "host_scaler.json",
        features_dir / "vocab" / "host_unigram.json",
        features_dir / "vocab" / "host_bigram.json",
    ]
    entries = {path.relative_to(features_dir).as_posix(): sha256_file(path) for path in paths}
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _checkpoint_document(
    *,
    config: TraceAnchorConfig,
    config_path: Path,
    channel: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    feature_spec: FeatureSpec,
    feature_manifest_hash: str,
    split_hash: str,
    git_commit: str,
    alpha: float,
    best_metric: float,
    epochs_without_improvement: int,
    amp_enabled: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "channel": channel,
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "rng_state": _rng_state(),
        "config": config.detector.model_dump(mode="json"),
        "config_sha256": config_hash(config, config_path),
        "feature_names": list(feature_spec.names),
        "feature_manifest_hash": feature_manifest_hash,
        "trigger_split_sha256": split_hash,
        "seed_value": seed,
        "git_commit": git_commit,
        "focal_alpha": alpha,
        "best_validation_scene_macro_pr_auc": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "amp_enabled": amp_enabled,
        "deterministic_algorithms": True,
        "allow_tf32": False,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _loader(
    dataset: CausalSceneDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )


@torch.no_grad()
def _validate(
    model: nn.Module,
    dataset: CausalSceneDataset,
    device: torch.device,
    *,
    batch_size: int,
    pin_memory: bool,
) -> dict[str, float | int]:
    model.eval()
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        workers=0,
        pin_memory=pin_memory,
        seed=0,
    )
    targets = []
    scores = []
    scene_indices = []
    for windows, labels, _mask, scenes, _rows in loader:
        logits = model(windows.to(device, non_blocking=pin_memory))
        targets.append(labels.numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
        scene_indices.append(scenes.numpy())
    return scene_macro_metrics(
        np.concatenate(targets),
        np.concatenate(scores),
        np.concatenate(scene_indices),
    )


def _model_paths(
    config: TraceAnchorConfig,
    config_path: Path,
    channel: str,
    seed: int,
    smoke_test: bool,
) -> tuple[Path, Path, Path, Path]:
    resolved = config.resolved_dict(config_path)
    artifacts = Path(resolved["paths"]["artifacts_dir"])
    if smoke_test:
        model_dir = artifacts / "smoke" / "models" / channel / f"seed_{seed}"
        log_path = artifacts / "smoke" / "logs" / f"{channel}_seed{seed}.json"
        marker = artifacts / "smoke" / "completion_markers" / f"{channel}_seed{seed}.done"
    else:
        model_dir = Path(resolved["paths"]["models_dir"]) / channel / f"seed_{seed}"
        log_path = Path(resolved["paths"]["logs_dir"]) / "detector" / f"{channel}_seed{seed}.json"
        marker = Path(resolved["paths"]["completion_markers_dir"]) / f"WP3_{channel}_seed{seed}.done"
    return model_dir / "best.pt", model_dir / "last.pt", log_path, marker


def train_detector(
    config: TraceAnchorConfig,
    config_path: Path,
    channel: str,
    seed: int,
    *,
    resume: bool = False,
    smoke_test: bool = False,
) -> dict[str, object]:
    if seed not in config.detector.training.seeds:
        raise ValueError(f"seed {seed} is not in the frozen detector seed list")
    if channel not in {"network", "host"}:
        raise ValueError(f"invalid detector channel: {channel}")
    resolved = config.resolved_dict(config_path)
    features_dir = Path(resolved["paths"]["features_dir"])
    best_path, last_path, log_path, marker_path = _model_paths(
        config, config_path, channel, seed, smoke_test
    )
    if resume and marker_path.exists() and best_path.exists():
        return json.loads(marker_path.read_text(encoding="utf-8"))

    set_determinism(seed)
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by project.yml but is unavailable; run outside the restricted sandbox"
        )
    spec = load_feature_spec(features_dir, channel)
    limit_train = 4 if smoke_test else None
    limit_validation = 2 if smoke_test else None
    train_scenes = load_scenes(
        features_dir, channel, "train", spec, limit=limit_train
    )
    validation_scenes = load_scenes(
        features_dir, channel, "validation", spec, limit=limit_validation
    )
    context = config.detector.common.context_seconds
    train_dataset = CausalSceneDataset(train_scenes, context, spec.padding_vector)
    validation_dataset = CausalSceneDataset(
        validation_scenes, context, spec.padding_vector
    )
    positives, negatives = class_counts(train_scenes)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"training split needs both classes, got positives={positives}, negatives={negatives}"
        )
    loss_config = config.detector.loss
    alpha = float(
        np.clip(
            negatives / (positives + negatives),
            loss_config.alpha_min,
            loss_config.alpha_max,
        )
    )
    model = build_model(config, channel, len(spec.names)).to(device)
    training = config.detector.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    max_abs_feature = max(
        float(np.max(np.abs(scene.features)))
        for scene in train_scenes + validation_scenes
    )
    # Robust scaling can legitimately retain extreme outliers. Values above
    # this bound can overflow during the initial FP16 projection, so those
    # runs remain deterministic FP32 rather than silently producing NaNs.
    amp_safe = max_abs_feature <= 1_000.0
    use_amp = bool(config.runtime.amp and device.type == "cuda" and amp_safe)
    amp_disabled_reason = (
        None
        if use_amp or not config.runtime.amp
        else f"max_abs_feature={max_abs_feature:.6g} exceeds fp16 safety bound 1000"
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    feature_hash = _feature_manifest_hash(features_dir)
    split_hash = sha256_file(
        Path(resolved["paths"]["splits_dir"]) / "trigger_split.parquet"
    )
    git_commit = _git_commit(Path(resolved["project"]["project_root"]))
    start_epoch = 1
    best_metric = float("-inf")
    epochs_without_improvement = 0
    epoch_records: list[dict[str, object]] = []
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint["channel"] != channel or int(checkpoint["seed"]) != seed:
            raise ValueError("resume checkpoint identity mismatch")
        if checkpoint["feature_manifest_hash"] != feature_hash:
            raise ValueError("feature manifest changed since checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        amp_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        _restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_validation_scene_macro_pr_auc"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if log_path.exists():
            epoch_records = list(json.loads(log_path.read_text(encoding="utf-8"))["epochs"])

    max_epochs = 1 if smoke_test else training.max_epochs
    process = psutil.Process()
    started = time.monotonic()
    stopped_early = False
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        loader = _loader(
            train_dataset,
            batch_size=training.batch_size,
            shuffle=True,
            workers=0 if smoke_test else training.num_workers,
            pin_memory=bool(training.pin_memory and device.type == "cuda"),
            seed=seed + epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0
        for batch_index, (windows, labels, mask, _scenes, _rows) in enumerate(loader):
            windows = windows.to(device, non_blocking=training.pin_memory)
            labels = labels.to(device, non_blocking=training.pin_memory)
            mask = mask.to(device, non_blocking=training.pin_memory)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model(windows)
                loss = focal_bce_with_logits(
                    logits,
                    labels,
                    mask,
                    alpha=alpha,
                    gamma=loss_config.gamma,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite {channel} loss at seed={seed}, epoch={epoch}, "
                    f"batch={batch_index}; max_abs_feature={max_abs_feature:.6g}, "
                    f"amp_enabled={use_amp}"
                )
            unscaled_loss = float(loss.detach().cpu())
            amp_scaler.scale(loss / training.gradient_accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % training.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training.gradient_clip_norm
                )
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += unscaled_loss * len(labels)
            total_examples += len(labels)

        validation = _validate(
            model,
            validation_dataset,
            device,
            batch_size=training.batch_size,
            pin_memory=bool(training.pin_memory and device.type == "cuda"),
        )
        metric = float(validation["scene_macro_pr_auc"])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        gpu_stats = {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "max_allocated_gib": 0.0,
        }
        if device.type == "cuda":
            gpu_stats = {
                "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                "max_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            }
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            **validation,
            "rss_gib": process.memory_info().rss / (1024**3),
            "gpu": gpu_stats,
            "elapsed_seconds": time.monotonic() - started,
            "amp_enabled": use_amp,
            "amp_disabled_reason": amp_disabled_reason,
        }
        epoch_records.append(record)
        document = _checkpoint_document(
            config=config,
            config_path=config_path,
            channel=channel,
            seed=seed,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=amp_scaler,
            feature_spec=spec,
            feature_manifest_hash=feature_hash,
            split_hash=split_hash,
            git_commit=git_commit,
            alpha=alpha,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            amp_enabled=use_amp,
        )
        if improved:
            _atomic_torch_save(best_path, document)
        _atomic_torch_save(last_path, document)
        atomic_write_json(
            log_path,
            {
                "schema_version": 1,
                "channel": channel,
                "seed": seed,
                "epochs": epoch_records,
            },
        )
        print(json.dumps({"channel": channel, "seed": seed, **record}), flush=True)
        if not smoke_test and epochs_without_improvement >= training.early_stopping_patience:
            stopped_early = True
            break

    if not best_path.exists():
        raise RuntimeError("training completed without a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu")
    summary = {
        "schema_version": 1,
        "channel": channel,
        "seed": seed,
        "smoke_test": smoke_test,
        "epochs_completed": len(epoch_records),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_scene_macro_pr_auc": float(
            best_checkpoint["best_validation_scene_macro_pr_auc"]
        ),
        "stopped_early": stopped_early,
        "positive_targets": positives,
        "negative_targets": negatives,
        "focal_alpha": alpha,
        "feature_count": len(spec.names),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "amp_enabled": use_amp,
        "amp_disabled_reason": amp_disabled_reason,
        "max_abs_feature": max_abs_feature,
        "deterministic_algorithms": True,
        "effective_batch_size": training.batch_size
        * training.gradient_accumulation_steps,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "feature_manifest_hash": feature_hash,
        "trigger_split_sha256": split_hash,
        "git_commit": git_commit,
    }
    atomic_write_json(marker_path, summary)
    if not smoke_test:
        markers_dir = Path(resolved["paths"]["completion_markers_dir"])
        expected = [
            markers_dir / f"WP3_{candidate_channel}_seed{candidate_seed}.done"
            for candidate_channel in ("network", "host")
            for candidate_seed in training.seeds
        ]
        if all(path.exists() for path in expected):
            atomic_write_json(
                markers_dir / "WP3_models.done",
                {
                    "schema_version": 1,
                    "models": [json.loads(path.read_text(encoding="utf-8")) for path in expected],
                },
            )
    return summary


__all__ = ["build_model", "set_determinism", "train_detector"]
