from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from traceanchor.config import TraceAnchorConfig, config_hash
from traceanchor.detector.metrics import scene_macro_metrics
from traceanchor.detector.training import build_model
from traceanchor.ingest.common import atomic_write_json, atomic_write_table, sha256_file


TRACKS = ("network", "host", "router")


@dataclass(frozen=True)
class Episode:
    token: str
    start: int
    end: int
    channels: tuple[str, ...]
    peak_network: float | None
    peak_host: float | None


ALERT_SCHEMA = pa.schema(
    [
        ("alert_id", pa.string()),
        ("scenario_token", pa.string()),
        ("channels", pa.list_(pa.string())),
        ("start_ts", pa.int64()),
        ("end_ts", pa.int64()),
        ("peak_score_network", pa.float32()),
        ("peak_score_host", pa.float32()),
        ("threshold_version", pa.string()),
        ("model_hashes", pa.list_(pa.string())),
    ]
)


GRID_SCHEMA = pa.schema(
    [
        ("track", pa.string()),
        ("tau_on", pa.float64()),
        ("tau_off", pa.float64()),
        ("merge_gap_seconds", pa.int64()),
        ("cooldown_seconds", pa.int64()),
        ("seed_count", pa.int64()),
        ("scenario_observations", pa.int64()),
        ("duration_seconds", pa.int64()),
        ("alert_count", pa.int64()),
        ("anchor_count", pa.int64()),
        ("recalled_anchors", pa.int64()),
        ("event_recall", pa.float64()),
        ("alerts_per_hour", pa.float64()),
        ("alerts_per_scenario", pa.float64()),
        ("median_detection_delay_seconds", pa.float64()),
        ("within_target_budget", pa.bool_()),
    ]
)


def aggregate_channel_episodes(
    token: str,
    seconds: list[int],
    scores: list[float],
    channel: str,
    *,
    tau_on: float,
    tau_off: float,
    merge_gap_seconds: int,
    cooldown_seconds: int,
    max_episode_seconds: int,
) -> list[Episode]:
    if tau_off > tau_on:
        raise ValueError("tau_off must not exceed tau_on")
    if channel not in {"network", "host"}:
        raise ValueError(f"invalid alert channel: {channel}")
    raw: list[Episode] = []
    start: int | None = None
    end: int | None = None
    peak = float("-inf")
    previous: int | None = None

    def close_active() -> None:
        nonlocal start, end, peak
        if start is None or end is None:
            return
        raw.append(
            Episode(
                token=token,
                start=start,
                end=end,
                channels=(channel,),
                peak_network=peak if channel == "network" else None,
                peak_host=peak if channel == "host" else None,
            )
        )
        start = None
        end = None
        peak = float("-inf")

    for second, score in zip(seconds, scores):
        second = int(second)
        score = float(score)
        discontinuity = previous is not None and second != previous + 1
        too_long = start is not None and second - start + 1 > max_episode_seconds
        if start is not None and (discontinuity or score < tau_off or too_long):
            close_active()
        if start is None and score >= tau_on:
            start = second
            end = second
            peak = score
        elif start is not None:
            end = second
            peak = max(peak, score)
        previous = second
    close_active()

    merged: list[Episode] = []
    for episode in raw:
        if merged:
            prior = merged[-1]
            gap = episode.start - prior.end - 1
            combined_length = episode.end - prior.start + 1
            if gap <= merge_gap_seconds and combined_length <= max_episode_seconds:
                merged[-1] = Episode(
                    token=token,
                    start=prior.start,
                    end=episode.end,
                    channels=(channel,),
                    peak_network=(
                        max(float(prior.peak_network), float(episode.peak_network))
                        if channel == "network"
                        else None
                    ),
                    peak_host=(
                        max(float(prior.peak_host), float(episode.peak_host))
                        if channel == "host"
                        else None
                    ),
                )
                continue
        merged.append(episode)
    kept = []
    for episode in merged:
        if kept and episode.start <= kept[-1].end + cooldown_seconds:
            continue
        kept.append(episode)
    return kept


def merge_router_episodes(episodes: list[Episode]) -> list[Episode]:
    output: list[Episode] = []
    for episode in sorted(episodes, key=lambda value: (value.start, value.end, value.channels)):
        if output and episode.start <= output[-1].end:
            prior = output[-1]
            output[-1] = Episode(
                token=prior.token,
                start=min(prior.start, episode.start),
                end=max(prior.end, episode.end),
                channels=tuple(sorted(set(prior.channels) | set(episode.channels))),
                peak_network=max(
                    [value for value in (prior.peak_network, episode.peak_network) if value is not None],
                    default=None,
                ),
                peak_host=max(
                    [value for value in (prior.peak_host, episode.peak_host) if value is not None],
                    default=None,
                ),
            )
        else:
            output.append(episode)
    return output


def _load_scores(
    config: TraceAnchorConfig, config_path: Path, split: str
) -> tuple[
    dict[int, dict[str, dict[str, tuple[list[int], list[float]]]]],
    dict[tuple[int, str], str],
]:
    resolved = config.resolved_dict(config_path)
    scores_dir = Path(resolved["paths"]["artifacts_dir"]) / "scores" / split
    values: dict[int, dict[str, dict[str, tuple[list[int], list[float]]]]] = {}
    model_hashes = {}
    for seed in config.detector.training.seeds:
        values[seed] = {}
        for channel in ("network", "host"):
            path = scores_dir / f"{channel}_seed{seed}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"score file missing: {path}")
            rows = pq.read_table(path).to_pylist()
            by_token: dict[str, tuple[list[int], list[float]]] = {}
            for row in rows:
                token = str(row["scenario_token"])
                seconds, scores = by_token.setdefault(token, ([], []))
                seconds.append(int(row["second_ts"]))
                scores.append(float(row["score"]))
                model_hashes[(seed, channel)] = str(row["model_hash"])
            values[seed][channel] = by_token
    return values, model_hashes


def _load_private_targets(
    config: TraceAnchorConfig, config_path: Path, split: str
) -> dict[str, dict[int, tuple[int, int]]]:
    resolved = config.resolved_dict(config_path)
    features_dir = Path(resolved["paths"]["features_dir"])
    targets: dict[str, dict[int, tuple[int, int]]] = {}
    for channel in ("network", "host"):
        for path in sorted((features_dir / channel / split).glob("*.parquet")):
            table = pq.read_table(
                path, columns=["second_ts", "label_private", "loss_mask"]
            )
            token_targets = targets.setdefault(path.stem, {})
            for second, label, mask in zip(
                table["second_ts"].to_pylist(),
                table["label_private"].to_pylist(),
                table["loss_mask"].to_pylist(),
            ):
                new_value = (1 if label == "positive_core" else 0, int(mask))
                old_value = token_targets.get(int(second), (0, 0))
                token_targets[int(second)] = (
                    max(old_value[0], new_value[0]),
                    max(old_value[1], new_value[1]),
                )
    return targets


def _episodes_for_track(
    score_values: dict[str, dict[str, tuple[list[int], list[float]]]],
    track: str,
    parameters: dict[str, float | int],
    max_episode_seconds: int,
) -> dict[str, list[Episode]]:
    tokens = sorted(set(score_values["network"]) | set(score_values["host"]))
    result = {}
    for token in tokens:
        channels = (track,) if track in {"network", "host"} else ("network", "host")
        episodes = []
        for channel in channels:
            seconds, scores = score_values[channel].get(token, ([], []))
            episodes.extend(
                aggregate_channel_episodes(
                    token,
                    seconds,
                    scores,
                    channel,
                    tau_on=float(parameters["tau_on"]),
                    tau_off=float(parameters["tau_off"]),
                    merge_gap_seconds=int(parameters["merge_gap_seconds"]),
                    cooldown_seconds=int(parameters["cooldown_seconds"]),
                    max_episode_seconds=max_episode_seconds,
                )
            )
        result[token] = merge_router_episodes(episodes) if track == "router" else episodes
    return result


def _episode_metrics(
    all_seed_episodes: list[dict[str, list[Episode]]],
    score_values: dict[int, dict[str, dict[str, tuple[list[int], list[float]]]]],
    targets: dict[str, dict[int, tuple[int, int]]],
    track: str,
) -> dict[str, float | int | None]:
    alert_count = 0
    anchor_count = 0
    recalled = 0
    delays = []
    duration = 0
    scenario_observations = 0
    for seed, episodes_by_token in zip(sorted(score_values), all_seed_episodes):
        channels = (track,) if track in {"network", "host"} else ("network", "host")
        tokens = sorted(set().union(*(set(score_values[seed][channel]) for channel in channels)))
        scenario_observations += len(tokens)
        for token in tokens:
            seconds = set()
            for channel in channels:
                seconds.update(score_values[seed][channel].get(token, ([], []))[0])
            duration += len(seconds)
            episodes = episodes_by_token.get(token, [])
            alert_count += len(episodes)
            anchors = sorted(
                second
                for second, (target, _mask) in targets.get(token, {}).items()
                if target == 1
            )
            anchor_count += len(anchors)
            for anchor in anchors:
                matches = [episode for episode in episodes if episode.end >= anchor]
                if matches:
                    match = min(matches, key=lambda episode: max(anchor, episode.start))
                    recalled += 1
                    delays.append(max(0, match.start - anchor))
    return {
        "seed_count": len(all_seed_episodes),
        "scenario_observations": scenario_observations,
        "duration_seconds": duration,
        "alert_count": alert_count,
        "anchor_count": anchor_count,
        "recalled_anchors": recalled,
        "event_recall": 0.0 if anchor_count == 0 else recalled / anchor_count,
        "alerts_per_hour": 0.0 if duration == 0 else alert_count * 3600.0 / duration,
        "alerts_per_scenario": (
            0.0 if scenario_observations == 0 else alert_count / scenario_observations
        ),
        "median_detection_delay_seconds": (
            None if not delays else float(statistics.median(delays))
        ),
    }


def _parameters(config: TraceAnchorConfig):
    grid = config.alerting.threshold_grid
    for tau_on, tau_off, merge_gap, cooldown in itertools.product(
        grid.tau_on,
        grid.tau_off,
        grid.merge_gap_seconds,
        grid.cooldown_seconds,
    ):
        if tau_off <= tau_on:
            yield {
                "tau_on": float(tau_on),
                "tau_off": float(tau_off),
                "merge_gap_seconds": int(merge_gap),
                "cooldown_seconds": int(cooldown),
            }


def _selection_key(row: dict[str, object]) -> tuple:
    delay = row["median_detection_delay_seconds"]
    return (
        -float(row["event_recall"]),
        float("inf") if delay is None else float(delay),
        float(row["tau_on"]) - float(row["tau_off"]),
        int(row["merge_gap_seconds"]),
        int(row["cooldown_seconds"]),
        -float(row["tau_on"]),
    )


def _alert_rows(
    episodes_by_token: dict[str, list[Episode]],
    *,
    track: str,
    seed: int,
    threshold_version: str,
    model_hashes: dict[tuple[int, str], str],
) -> list[dict[str, object]]:
    output = []
    for token, episodes in sorted(episodes_by_token.items()):
        for episode in episodes:
            material = (
                f"{track}|{seed}|{token}|{episode.start}|{episode.end}|"
                f"{','.join(episode.channels)}|{threshold_version}"
            )
            alert_id = "al_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
            output.append(
                {
                    "alert_id": alert_id,
                    "scenario_token": token,
                    "channels": list(episode.channels),
                    "start_ts": episode.start,
                    "end_ts": episode.end,
                    "peak_score_network": episode.peak_network,
                    "peak_score_host": episode.peak_host,
                    "threshold_version": threshold_version,
                    "model_hashes": [
                        model_hashes[(seed, channel)] for channel in episode.channels
                    ],
                }
            )
    return output


def _materialize_alerts(
    config: TraceAnchorConfig,
    config_path: Path,
    split: str,
    selected: dict[str, object],
    score_values: dict[int, dict[str, dict[str, tuple[list[int], list[float]]]]],
    model_hashes: dict[tuple[int, str], str],
) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    output_dir = Path(resolved["paths"]["alerts_dir"]) / split
    threshold_version = str(selected["threshold_version"])
    results = []
    for track in TRACKS:
        parameters = selected["tracks"][track]["parameters"]
        for seed in config.detector.training.seeds:
            episodes = _episodes_for_track(
                score_values[seed],
                track,
                parameters,
                config.alerting.max_episode_seconds,
            )
            rows = _alert_rows(
                episodes,
                track=track,
                seed=seed,
                threshold_version=threshold_version,
                model_hashes=model_hashes,
            )
            path = output_dir / f"{track}_seed{seed}.parquet"
            atomic_write_table(path, pa.Table.from_pylist(rows, schema=ALERT_SCHEMA))
            results.append(
                {
                    "track": track,
                    "seed": seed,
                    "alerts": len(rows),
                    "sha256": sha256_file(path),
                }
            )
    return {"split": split, "files": results}


def calibrate_alerts(
    config: TraceAnchorConfig, config_path: Path
) -> dict[str, object]:
    split = config.alerting.calibration_split
    score_values, model_hashes = _load_scores(config, config_path, split)
    targets = _load_private_targets(config, config_path, split)
    rows = []
    for track in TRACKS:
        for parameters in _parameters(config):
            all_seed_episodes = [
                _episodes_for_track(
                    score_values[seed],
                    track,
                    parameters,
                    config.alerting.max_episode_seconds,
                )
                for seed in sorted(score_values)
            ]
            metrics = _episode_metrics(
                all_seed_episodes, score_values, targets, track
            )
            rows.append(
                {
                    "track": track,
                    **parameters,
                    **metrics,
                    "within_target_budget": float(metrics["alerts_per_hour"])
                    <= config.alerting.target_alerts_per_hour,
                }
            )
    resolved = config.resolved_dict(config_path)
    alerts_dir = Path(resolved["paths"]["alerts_dir"])
    grid_path = alerts_dir / "calibration_grid.parquet"
    atomic_write_table(grid_path, pa.Table.from_pylist(rows, schema=GRID_SCHEMA))
    track_selections = {}
    for track in TRACKS:
        candidates = [row for row in rows if row["track"] == track]
        within = [row for row in candidates if row["within_target_budget"]]
        pool = within if within else candidates
        chosen = min(pool, key=_selection_key)
        track_selections[track] = {
            "parameters": {
                key: chosen[key]
                for key in (
                    "tau_on",
                    "tau_off",
                    "merge_gap_seconds",
                    "cooldown_seconds",
                )
            },
            "validation_metrics": {
                key: chosen[key]
                for key in (
                    "event_recall",
                    "alerts_per_hour",
                    "alerts_per_scenario",
                    "median_detection_delay_seconds",
                    "alert_count",
                    "anchor_count",
                    "recalled_anchors",
                )
            },
            "budget_satisfied": bool(chosen["within_target_budget"]),
        }
    version_payload = {
        "config_sha256": config_hash(config, config_path),
        "calibration_split": split,
        "target_alerts_per_hour": config.alerting.target_alerts_per_hour,
        "tracks": track_selections,
        "grid_sha256": sha256_file(grid_path),
    }
    canonical = json.dumps(version_payload, sort_keys=True, separators=(",", ":"))
    threshold_version = "thr_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    selected = {
        "schema_version": 1,
        "threshold_version": threshold_version,
        **version_payload,
        "selection_priority": [
            "satisfy_alert_budget",
            "maximize_event_recall",
            "minimize_detection_delay",
            "prefer_simpler_parameters",
        ],
        "matching_rule": (
            "an anchor is recalled by the first same-scene episode whose end is "
            "at or after the anchor; delay=max(0, episode_start-anchor)"
        ),
    }
    selected_path = alerts_dir / "thresholds.json"
    atomic_write_json(selected_path, selected)
    validation_outputs = _materialize_alerts(
        config,
        config_path,
        split,
        selected,
        score_values,
        model_hashes,
    )
    marker = Path(resolved["paths"]["completion_markers_dir"]) / "WP3_calibration.done"
    summary = {
        "schema_version": 1,
        "threshold_version": threshold_version,
        "grid_candidates": len(rows),
        "grid_sha256": sha256_file(grid_path),
        "selected_sha256": sha256_file(selected_path),
        "tracks": track_selections,
        "validation_alerts": validation_outputs,
    }
    atomic_write_json(marker, summary)
    return summary


def _censored_metrics(
    score_values: dict[str, dict[str, tuple[list[int], list[float]]]],
    targets: dict[str, dict[int, tuple[int, int]]],
    track: str,
    threshold: float,
) -> dict[str, float | int]:
    channels = (track,) if track in {"network", "host"} else ("network", "host")
    all_targets = []
    all_scores = []
    all_scenes = []
    token_set = sorted(set().union(*(set(score_values[channel]) for channel in channels)))
    for scene_index, token in enumerate(token_set):
        by_second: dict[int, list[float]] = {}
        for channel in channels:
            seconds, scores = score_values[channel].get(token, ([], []))
            for second, score in zip(seconds, scores):
                by_second.setdefault(int(second), []).append(float(score))
        for second, candidate_scores in sorted(by_second.items()):
            target, mask = targets.get(token, {}).get(second, (0, 0))
            if mask != 1:
                continue
            all_targets.append(target)
            all_scores.append(max(candidate_scores))
            all_scenes.append(scene_index)
    if not all_targets:
        raise ValueError(f"no censored labeled rows for {track}")
    return scene_macro_metrics(
        np.asarray(all_targets, dtype=np.float32),
        np.asarray(all_scores, dtype=np.float32),
        np.asarray(all_scenes, dtype=np.int64),
        threshold=threshold,
    )


def _configured_causality_checks(config: TraceAnchorConfig) -> dict[str, bool]:
    torch.manual_seed(20260726)
    results = {}
    for channel, input_dim in (("network", 7), ("host", 11)):
        model = build_model(config, channel, input_dim).eval()
        original = torch.randn(2, 40, input_dim)
        perturbed = original.clone()
        perturbed[:, 21:] = torch.randn_like(perturbed[:, 21:]) * 1000
        with torch.no_grad():
            before = model.forward_sequence(original)
            after = model.forward_sequence(perturbed)
        results[channel] = bool(torch.equal(before[:, :21], after[:, :21]))
    return results


def evaluate_alerts(
    config: TraceAnchorConfig, config_path: Path
) -> dict[str, object]:
    resolved = config.resolved_dict(config_path)
    alerts_dir = Path(resolved["paths"]["alerts_dir"])
    selected_path = alerts_dir / "thresholds.json"
    if not selected_path.exists():
        raise FileNotFoundError("Validation thresholds are not frozen")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["calibration_split"] != "validation":
        raise ValueError("threshold file was not calibrated on Validation")
    split = "test"
    score_values, model_hashes = _load_scores(config, config_path, split)
    targets = _load_private_targets(config, config_path, split)
    alert_outputs = _materialize_alerts(
        config, config_path, split, selected, score_values, model_hashes
    )
    track_results = {}
    for track in TRACKS:
        parameters = selected["tracks"][track]["parameters"]
        per_seed = []
        all_seed_episodes = []
        for seed in sorted(score_values):
            episodes = _episodes_for_track(
                score_values[seed],
                track,
                parameters,
                config.alerting.max_episode_seconds,
            )
            all_seed_episodes.append(episodes)
            episode_metrics = _episode_metrics(
                [episodes], {seed: score_values[seed]}, targets, track
            )
            censored = _censored_metrics(
                score_values[seed],
                targets,
                track,
                float(parameters["tau_on"]),
            )
            censored_positive_recall = censored.pop("event_recall")
            per_seed.append(
                {
                    "seed": seed,
                    **episode_metrics,
                    **censored,
                    "censored_positive_second_recall": censored_positive_recall,
                }
            )
        aggregate = _episode_metrics(
            all_seed_episodes, score_values, targets, track
        )
        aggregate.update(
            {
                "scene_macro_pr_auc": float(
                    np.mean([row["scene_macro_pr_auc"] for row in per_seed])
                ),
                "scene_macro_f1": float(
                    np.mean([row["scene_macro_f1"] for row in per_seed])
                ),
                "effective_scenes": sum(int(row["effective_scenes"]) for row in per_seed),
                "labeled_windows": sum(int(row["labeled_windows"]) for row in per_seed),
            }
        )
        track_results[track] = {
            "parameters_from_validation": parameters,
            "aggregate": aggregate,
            "per_seed": per_seed,
        }
    grid_rows = pq.read_table(alerts_dir / "calibration_grid.parquet").to_pylist()
    recall_at_budget = {}
    for track in TRACKS:
        recall_at_budget[track] = {}
        rows = [row for row in grid_rows if row["track"] == track]
        for budget in (1.0, 5.0, 10.0):
            eligible = [row for row in rows if float(row["alerts_per_hour"]) <= budget]
            recall_at_budget[track][f"recall_at_{int(budget)}_alerts_per_hour"] = (
                0.0 if not eligible else max(float(row["event_recall"]) for row in eligible)
            )
    report = {
        "schema_version": 1,
        "split": split,
        "threshold_version": selected["threshold_version"],
        "calibration_split": selected["calibration_split"],
        "test_recalibrated": False,
        "matching_rule": selected["matching_rule"],
        "tracks": track_results,
        "validation_recall_at_alert_budgets": recall_at_budget,
        "alert_outputs": alert_outputs,
    }
    metrics_dir = Path(resolved["paths"]["metrics_dir"])
    report_path = metrics_dir / "trigger_evaluation.json"
    atomic_write_json(report_path, report)

    completion_dir = Path(resolved["paths"]["completion_markers_dir"])
    required_markers = [
        completion_dir / name
        for name in (
            "WP3_models.done",
            "WP3_scores_validation.done",
            "WP3_calibration.done",
            "WP3_scores_test.done",
        )
    ]
    errors = [f"missing {path.name}" for path in required_markers if not path.exists()]
    expected_models = {
        (channel, seed)
        for channel in ("network", "host")
        for seed in config.detector.training.seeds
    }
    checkpoint_required = {
        "config",
        "feature_manifest_hash",
        "trigger_split_sha256",
        "seed",
        "epoch",
        "git_commit",
        "model_state_dict",
    }
    finite_training_logs = True
    for channel, seed in sorted(expected_models):
        model_marker = completion_dir / f"WP3_{channel}_seed{seed}.done"
        if not model_marker.exists():
            errors.append(f"missing model marker {model_marker.name}")
            continue
        summary = json.loads(model_marker.read_text(encoding="utf-8"))
        checkpoint_path = Path(summary["best_checkpoint"])
        if not checkpoint_path.exists():
            errors.append(f"missing checkpoint {checkpoint_path}")
            continue
        if sha256_file(checkpoint_path) != summary["best_checkpoint_sha256"]:
            errors.append(f"checkpoint hash mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing_fields = checkpoint_required.difference(checkpoint)
        if missing_fields:
            errors.append(
                f"checkpoint metadata missing for {channel} seed {seed}: {sorted(missing_fields)}"
            )
        if checkpoint.get("channel") != channel or int(checkpoint.get("seed", -1)) != seed:
            errors.append(f"checkpoint identity mismatch for {channel} seed {seed}")
        log_path = Path(resolved["paths"]["logs_dir"]) / "detector" / f"{channel}_seed{seed}.json"
        if not log_path.exists():
            errors.append(f"missing training log {log_path.name}")
            finite_training_logs = False
            continue
        log = json.loads(log_path.read_text(encoding="utf-8"))
        losses = [float(epoch["train_loss"]) for epoch in log["epochs"]]
        if not losses or not all(math.isfinite(value) for value in losses):
            errors.append(f"non-finite or empty training loss log: {log_path.name}")
            finite_training_logs = False

    checkpoint_reload_exact = True
    score_schemas_public = True
    for score_split in ("validation", "test"):
        score_marker = completion_dir / f"WP3_scores_{score_split}.done"
        if not score_marker.exists():
            checkpoint_reload_exact = False
            continue
        score_summary = json.loads(score_marker.read_text(encoding="utf-8"))
        identities = {
            (str(item["channel"]), int(item["seed"]))
            for item in score_summary["models"]
        }
        if identities != expected_models:
            errors.append(f"score identities incomplete for {score_split}")
        if not all(bool(item["checkpoint_reload_exact"]) for item in score_summary["models"]):
            errors.append(f"checkpoint reload mismatch in {score_split} scoring")
            checkpoint_reload_exact = False
        scores_dir = Path(resolved["paths"]["artifacts_dir"]) / "scores" / score_split
        for channel, seed in expected_models:
            score_path = scores_dir / f"{channel}_seed{seed}.parquet"
            names = {name.lower() for name in pq.read_schema(score_path).names}
            if names.intersection({"label", "gold", "resource", "family", "cve", "role"}):
                errors.append(f"private field in score schema: {score_path.name}")
                score_schemas_public = False

    grid_path = alerts_dir / "calibration_grid.parquet"
    expected_grid_rows = len(list(_parameters(config))) * len(TRACKS)
    actual_grid_rows = pq.read_metadata(grid_path).num_rows
    if actual_grid_rows != expected_grid_rows:
        errors.append(
            f"calibration grid has {actual_grid_rows} rows, expected {expected_grid_rows}"
        )
    causality_checks = _configured_causality_checks(config)
    if not all(causality_checks.values()):
        errors.append(f"configured causality checks failed: {causality_checks}")
    for track in TRACKS:
        for seed in config.detector.training.seeds:
            path = alerts_dir / split / f"{track}_seed{seed}.parquet"
            if not path.exists():
                errors.append(f"missing alert output {path.name}")
            elif pq.read_schema(path) != ALERT_SCHEMA:
                errors.append(f"alert schema mismatch: {path.name}")
            else:
                versions = set(
                    pq.read_table(path, columns=["threshold_version"])[
                        "threshold_version"
                    ].to_pylist()
                )
                if versions and versions != {selected["threshold_version"]}:
                    errors.append(f"Test threshold version mismatch: {path.name}")
    qa = {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "six_seed_models": (Path(resolved["paths"]["completion_markers_dir"]) / "WP3_models.done").exists(),
        "validation_only_calibration": selected["calibration_split"] == "validation",
        "test_recalibrated": False,
        "checkpoint_reload_exact": checkpoint_reload_exact,
        "finite_training_logs": finite_training_logs,
        "configured_future_perturbation_checks": causality_checks,
        "unit_test_source": "tests/unit/test_detector.py",
        "score_schemas_public": score_schemas_public,
        "calibration_grid_rows": actual_grid_rows,
        "expected_calibration_grid_rows": expected_grid_rows,
        "resource_detector_input": False,
        "learned_fusion": False,
        "trigger_evaluation_sha256": sha256_file(report_path),
    }
    qa_path = Path(resolved["paths"]["reports_dir"]) / "WP03_qa.json" if "reports_dir" in resolved["paths"] else Path(resolved["paths"]["artifacts_dir"]) / "reports" / "WP03_qa.json"
    atomic_write_json(qa_path, qa)
    if errors:
        raise ValueError(f"WP3 hard gate failed: {errors}")
    marker = Path(resolved["paths"]["completion_markers_dir"]) / "WP3.done"
    completion = {
        "schema_version": 1,
        "config_sha256": config_hash(config, config_path),
        "threshold_version": selected["threshold_version"],
        "qa_sha256": sha256_file(qa_path),
        "trigger_evaluation_sha256": sha256_file(report_path),
    }
    atomic_write_json(marker, completion)
    return report


__all__ = [
    "ALERT_SCHEMA",
    "Episode",
    "aggregate_channel_episodes",
    "calibrate_alerts",
    "evaluate_alerts",
    "merge_router_episodes",
]
