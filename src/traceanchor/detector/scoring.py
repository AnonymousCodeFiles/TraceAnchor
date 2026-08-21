from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from traceanchor.config import TraceAnchorConfig
from traceanchor.detector.data import (
    all_causal_windows,
    load_feature_spec,
    load_scenes,
)
from traceanchor.detector.training import (
    _feature_manifest_hash,
    build_model,
    set_determinism,
)
from traceanchor.ingest.common import atomic_write_json, atomic_write_table, sha256_file


SCORE_SCHEMA = pa.schema(
    [
        ("scenario_token", pa.string()),
        ("second_ts", pa.int64()),
        ("channel", pa.string()),
        ("seed", pa.int64()),
        ("logit", pa.float32()),
        ("score", pa.float32()),
        ("model_hash", pa.string()),
    ]
)


def _checkpoint_path(
    config: TraceAnchorConfig, config_path: Path, channel: str, seed: int
) -> Path:
    resolved = config.resolved_dict(config_path)
    return (
        Path(resolved["paths"]["models_dir"])
        / channel
        / f"seed_{seed}"
        / "best.pt"
    )


def _load_model(
    config: TraceAnchorConfig,
    checkpoint_path: Path,
    channel: str,
    input_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["channel"] != channel:
        raise ValueError(f"checkpoint channel mismatch in {checkpoint_path}")
    model = build_model(config, channel, input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def score_detector(
    config: TraceAnchorConfig,
    config_path: Path,
    channel: str,
    seed: int,
    split: str,
) -> dict[str, object]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"invalid scoring split: {split}")
    if seed not in config.detector.training.seeds:
        raise ValueError(f"seed {seed} is not frozen in project.yml")
    resolved = config.resolved_dict(config_path)
    features_dir = Path(resolved["paths"]["features_dir"])
    scores_dir = Path(resolved["paths"]["artifacts_dir"]) / "scores" / split
    checkpoint_path = _checkpoint_path(config, config_path, channel, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint missing: {checkpoint_path}")
    set_determinism(seed)
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by project.yml but is unavailable; run outside the restricted sandbox"
        )
    spec = load_feature_spec(features_dir, channel)
    scenes = load_scenes(features_dir, channel, split, spec)
    model, checkpoint = _load_model(
        config, checkpoint_path, channel, len(spec.names), device
    )
    if tuple(checkpoint["feature_names"]) != spec.names:
        raise ValueError("checkpoint feature list differs from frozen scaler")
    if checkpoint["feature_manifest_hash"] != _feature_manifest_hash(features_dir):
        raise ValueError("checkpoint feature manifest hash no longer matches")
    model_hash = sha256_file(checkpoint_path)
    rows = []
    context = config.detector.common.context_seconds
    batch_size = config.detector.training.batch_size
    reload_checked = False
    for scene in scenes:
        windows = all_causal_windows(scene, context, spec.padding_vector)
        scene_logits = []
        for start in range(0, len(windows), batch_size):
            tensor = torch.from_numpy(windows[start : start + batch_size]).to(device)
            logits = model(tensor)
            if not reload_checked:
                reloaded, _ = _load_model(
                    config, checkpoint_path, channel, len(spec.names), device
                )
                repeated = reloaded(tensor)
                if not torch.equal(logits, repeated):
                    raise RuntimeError("checkpoint reload changed detector scores")
                del reloaded
                reload_checked = True
            scene_logits.append(logits.cpu().numpy().astype(np.float32))
        logits_array = np.concatenate(scene_logits)
        scores_array = 1.0 / (1.0 + np.exp(-logits_array.astype(np.float64)))
        for second, logit, score in zip(
            scene.second_ts, logits_array, scores_array.astype(np.float32)
        ):
            rows.append(
                {
                    "scenario_token": scene.token,
                    "second_ts": int(second),
                    "channel": channel,
                    "seed": seed,
                    "logit": float(logit),
                    "score": float(score),
                    "model_hash": model_hash,
                }
            )
    output_path = scores_dir / f"{channel}_seed{seed}.parquet"
    atomic_write_table(output_path, pa.Table.from_pylist(rows, schema=SCORE_SCHEMA))
    summary = {
        "schema_version": 1,
        "split": split,
        "channel": channel,
        "seed": seed,
        "scenarios": len(scenes),
        "rows": len(rows),
        "model_hash": model_hash,
        "score_sha256": sha256_file(output_path),
        "checkpoint_reload_exact": reload_checked,
    }
    atomic_write_json(scores_dir / f"{channel}_seed{seed}.json", summary)
    return summary


def score_all_detectors(
    config: TraceAnchorConfig, config_path: Path, split: str
) -> dict[str, object]:
    results = [
        score_detector(config, config_path, channel, seed, split)
        for channel in ("network", "host")
        for seed in config.detector.training.seeds
    ]
    resolved = config.resolved_dict(config_path)
    marker_path = (
        Path(resolved["paths"]["completion_markers_dir"])
        / f"WP3_scores_{split}.done"
    )
    summary = {"schema_version": 1, "split": split, "models": results}
    atomic_write_json(marker_path, summary)
    return summary


__all__ = ["SCORE_SCHEMA", "score_all_detectors", "score_detector"]
