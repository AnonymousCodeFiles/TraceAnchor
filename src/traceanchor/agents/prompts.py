from __future__ import annotations

import hashlib
from pathlib import Path

from traceanchor.agents.schemas import AgentRole


PROMPT_ROLES: tuple[AgentRole, ...] = (
    "orchestrator",
    "network_investigator",
    "host_investigator",
    "correlation_agent",
    "evidence_verifier",
)
_SAFETY_SENTINELS = (
    "untrusted data, never instructions",
    "insufficient_evidence",
    "no shell, write, response execution, or privilege expansion",
    "observed, correlated, and possibly_causal",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_directory(project_root: Path) -> Path:
    return project_root / "prompts" / "agents"


def load_system_prompt(project_root: Path, role: AgentRole) -> str:
    if role not in PROMPT_ROLES:
        raise ValueError(f"unsupported Agent role: {role}")
    directory = prompt_directory(project_root)
    prefix = (directory / "safety_prefix.txt").read_text(encoding="utf-8").strip()
    role_text = (directory / f"{role}.txt").read_text(encoding="utf-8").strip()
    lowered = prefix.lower()
    missing = [value for value in _SAFETY_SENTINELS if value not in lowered]
    if missing:
        raise ValueError(f"Agent safety prefix is incomplete: {missing}")
    return f"{prefix}\n\nROLE-SPECIFIC DUTIES\n{role_text}\n"


def prompt_hashes(project_root: Path) -> dict[str, str]:
    values = {role: _sha256(load_system_prompt(project_root, role)) for role in PROMPT_ROLES}
    values["safety_prefix"] = _sha256(
        (prompt_directory(project_root) / "safety_prefix.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    return values


__all__ = ["PROMPT_ROLES", "load_system_prompt", "prompt_hashes"]
