from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from traceanchor.config import FORBIDDEN_AGENT_COLUMNS


@dataclass(frozen=True)
class FeatureSpec:
    names: tuple[str, ...]
    padding_vector: np.ndarray


@dataclass(frozen=True)
class SceneData:
    token: str
    second_ts: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray


def feature_paths(
    features_dir: Path, channel: str, split: str, *, limit: int | None = None
) -> list[Path]:
    if channel not in {"network", "host"}:
        raise ValueError(f"invalid detector channel: {channel}")
    paths = sorted((features_dir / channel / split).glob("*.parquet"))
    return paths if limit is None else paths[:limit]


def load_feature_spec(features_dir: Path, channel: str) -> FeatureSpec:
    scaler = json.loads(
        (features_dir / "scalers" / f"{channel}_scaler.json").read_text(
            encoding="utf-8"
        )
    )
    names = tuple(str(value) for value in scaler["feature_names"])
    forbidden = {value.lower() for value in FORBIDDEN_AGENT_COLUMNS} | {"resource"}
    leaked = [name for name in names if any(term in name.lower() for term in forbidden)]
    if leaked:
        raise ValueError(f"forbidden detector feature names: {leaked}")
    padding = np.zeros(len(names), dtype=np.float32)
    if "missing" in names:
        index = names.index("missing")
        median = float(scaler["median"][index])
        iqr = float(scaler["iqr"][index])
        padding[index] = (1.0 - median) / iqr
    return FeatureSpec(names=names, padding_vector=padding)


def load_scenes(
    features_dir: Path,
    channel: str,
    split: str,
    spec: FeatureSpec,
    *,
    limit: int | None = None,
) -> list[SceneData]:
    scenes = []
    for path in feature_paths(features_dir, channel, split, limit=limit):
        columns = [
            "scenario_token",
            "second_ts",
            *spec.names,
            "label_private",
            "loss_mask",
        ]
        table = pq.read_table(path, columns=columns)
        tokens = table["scenario_token"].to_pylist()
        if not tokens or any(str(value) != path.stem for value in tokens):
            raise ValueError(f"scenario token mismatch in {path}")
        matrix = np.column_stack(
            [
                np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float32)
                for name in spec.names
            ]
        )
        labels = np.asarray(
            [1.0 if value == "positive_core" else 0.0 for value in table["label_private"].to_pylist()],
            dtype=np.float32,
        )
        scenes.append(
            SceneData(
                token=path.stem,
                second_ts=np.asarray(
                    table["second_ts"].to_numpy(zero_copy_only=False), dtype=np.int64
                ),
                features=matrix,
                targets=labels,
                loss_mask=np.asarray(
                    table["loss_mask"].to_numpy(zero_copy_only=False), dtype=np.int8
                ),
            )
        )
    if not scenes:
        raise FileNotFoundError(f"no {channel} feature shards for split {split}")
    return scenes


class CausalSceneDataset(Dataset):
    """Labeled causal windows whose history never crosses a scene boundary."""

    def __init__(
        self,
        scenes: list[SceneData],
        context_seconds: int,
        padding_vector: np.ndarray,
    ) -> None:
        self.scenes = scenes
        self.context_seconds = context_seconds
        self.padding_vector = np.asarray(padding_vector, dtype=np.float32)
        self.index = [
            (scene_index, row_index)
            for scene_index, scene in enumerate(scenes)
            for row_index, mask in enumerate(scene.loss_mask)
            if int(mask) == 1
        ]
        if not self.index:
            raise ValueError("dataset contains no labeled target seconds")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        scene_index, row_index = self.index[index]
        scene = self.scenes[scene_index]
        start = max(0, row_index - self.context_seconds + 1)
        observed = scene.features[start : row_index + 1]
        padding_rows = self.context_seconds - len(observed)
        if padding_rows:
            padding = np.repeat(self.padding_vector[None, :], padding_rows, axis=0)
            window = np.concatenate((padding, observed), axis=0)
        else:
            window = observed
        return (
            torch.from_numpy(np.ascontiguousarray(window)),
            torch.tensor(scene.targets[row_index], dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(scene_index, dtype=torch.int64),
            torch.tensor(row_index, dtype=torch.int64),
        )


def class_counts(scenes: list[SceneData]) -> tuple[int, int]:
    positives = sum(
        int(np.logical_and(scene.targets == 1.0, scene.loss_mask == 1).sum())
        for scene in scenes
    )
    negatives = sum(
        int(np.logical_and(scene.targets == 0.0, scene.loss_mask == 1).sum())
        for scene in scenes
    )
    return positives, negatives


def all_causal_windows(
    scene: SceneData, context_seconds: int, padding_vector: np.ndarray
) -> np.ndarray:
    """Materialize every causal target window for one scene, with reset padding."""

    output = np.empty(
        (len(scene.second_ts), context_seconds, scene.features.shape[1]),
        dtype=np.float32,
    )
    output[:] = np.asarray(padding_vector, dtype=np.float32)
    for row_index in range(len(scene.second_ts)):
        start = max(0, row_index - context_seconds + 1)
        observed = scene.features[start : row_index + 1]
        output[row_index, -len(observed) :] = observed
    return output


__all__ = [
    "CausalSceneDataset",
    "FeatureSpec",
    "SceneData",
    "all_causal_windows",
    "class_counts",
    "feature_paths",
    "load_feature_spec",
    "load_scenes",
]
