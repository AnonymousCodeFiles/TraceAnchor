from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


FORBIDDEN_AGENT_COLUMNS = {
    "family",
    "cve",
    "role",
    "exploit",
    "exploit_name",
    "exploit_time",
    "image",
    "split",
    "label",
    "gold",
}


class Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class ProjectConfig(Section):
    name: Literal["traceanchor"]
    project_root: Path
    timezone: Literal["UTC"]
    global_seed: int


class RuntimeConfig(Section):
    python: Literal["3.10"]
    pytorch: Literal["2.1.2"]
    cuda: Literal["11.8"]
    cudnn: Literal["8.7"]
    device: str = Field(pattern=r"^(cuda:\d+|cpu)$")
    deterministic_algorithms: Literal[True]
    allow_tf32: Literal[False]
    amp: bool = True


class PathsConfig(Section):
    raw_data_root: Path
    example_scenario: Path
    artifacts_dir: Path
    manifests_dir: Path
    completion_markers_dir: Path


class IngestionConfig(Section):
    raw_read_only: Literal[True]
    require_four_files: Literal[True]
    sha256_raw_files: Literal[True]
    max_ram_gib: int = Field(gt=0, le=16)
    payload_mode: Literal["metadata_only"]
    payload_preview_bytes: Literal[0]
    atomic_writes: Literal[True]
    resume: Literal[True]


class PrivacyConfig(Section):
    forbidden_agent_columns: list[str]
    hide_original_paths: Literal[True]
    external_llm_payload_allowed: bool
    fail_on_leakage: Literal[True]

    @model_validator(mode="after")
    def require_all_forbidden_columns(self) -> "PrivacyConfig":
        missing = FORBIDDEN_AGENT_COLUMNS.difference(self.forbidden_agent_columns)
        if missing:
            raise ValueError(f"forbidden_agent_columns missing: {sorted(missing)}")
        return self


class TriggerSplitConfig(Section):
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    group_by: Literal["scenario_uid"]
    gold_reserved_to_test: Literal[True]

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> "TriggerSplitConfig":
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"trigger split ratios sum to {total}, expected 1.0")
        return self


class GoldSplitConfig(Section):
    scenarios_per_family: Literal[3]
    total_scenarios: Literal[45]


class SplitsConfig(Section):
    trigger: TriggerSplitConfig
    gold: GoldSplitConfig
    agent_development_cases: Literal[15]
    agent_test_cases: Literal[30]
    require_protocol_freeze_before_agent_test: Literal[True]


class LabelsConfig(Section):
    unknown_policy: Literal["censored_no_loss"]
    never_treat_unknown_as_negative: Literal[True]


class WindowingConfig(Section):
    interval_seconds: Literal[1]
    context_seconds: Literal[32]
    causal: Literal[True]
    preserve_scene_order: Literal[True]
    reset_between_scenes: Literal[True]


class ResourceFeatureConfig(Section):
    detector_input: Literal[False]
    evidence_only: Literal[True]


class HostFeatureConfig(Section):
    unigram_vocab_max: int = Field(gt=0, le=256)
    bigram_vocab_max: int = Field(gt=0, le=2048)
    include_oov_bucket: Literal[True]


class FeaturesConfig(Section):
    host: HostFeatureConfig
    resource: ResourceFeatureConfig


class DetectorCommonConfig(Section):
    context_seconds: Literal[32]
    input_projection_dim: Literal[64]
    channels: Literal[64]
    kernel_size: Literal[3]
    dilations: tuple[Literal[1], Literal[2], Literal[4], Literal[8]]
    dropout: float = Field(ge=0.0, lt=1.0)


class DetectorHostConfig(Section):
    pre_projection_hidden_dim: Literal[128]


class DetectorLossConfig(Section):
    name: Literal["class_balanced_focal_bce"]
    gamma: float = Field(ge=0.0)
    alpha_source: Literal["train_class_frequency"]
    alpha_min: float = Field(ge=0.0, le=1.0)
    alpha_max: float = Field(ge=0.0, le=1.0)
    apply_loss_mask: Literal[True]

    @model_validator(mode="after")
    def alpha_bounds_are_ordered(self) -> "DetectorLossConfig":
        if self.alpha_min > self.alpha_max:
            raise ValueError("detector loss alpha_min exceeds alpha_max")
        return self


class DetectorTrainingConfig(Section):
    seeds: tuple[Literal[42], Literal[123], Literal[456]]
    optimizer: Literal["adamw"]
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    gradient_clip_norm: float = Field(gt=0.0)
    max_epochs: int = Field(gt=0, le=50)
    early_stopping_patience: int = Field(gt=0)
    checkpoint_metric: Literal["validation_scene_macro_pr_auc"]
    num_workers: int = Field(ge=0)
    pin_memory: bool


class DetectorConfig(Section):
    architecture: Literal["dual_stream_causal_tcn"]
    no_parameter_sharing: Literal[True]
    no_learned_fusion: Literal[True]
    common: DetectorCommonConfig
    host: DetectorHostConfig
    loss: DetectorLossConfig
    training: DetectorTrainingConfig


class ThresholdGridConfig(Section):
    tau_on: list[float] = Field(min_length=1)
    tau_off: list[float] = Field(min_length=1)
    merge_gap_seconds: list[int] = Field(min_length=1)
    cooldown_seconds: list[int] = Field(min_length=1)


class AlertingConfig(Section):
    calibration_split: Literal["validation"]
    target_alerts_per_hour: float = Field(gt=0.0)
    threshold_grid: ThresholdGridConfig
    router: Literal["deterministic_or_merge"]
    max_episode_seconds: int = Field(gt=0, le=120)


class EvidenceStoreConfig(Section):
    agent_database_read_only: Literal[True]
    attach_evaluator_to_agent: Literal[False]
    max_query_time_range_seconds: int = Field(gt=0, le=180)
    max_records_per_tool_call: int = Field(gt=0, le=200)


class GoldAnnotationConfig(Section):
    double_annotated_cases_min: int = Field(ge=20, le=45)
    evidence_f1_agreement_gate: float = Field(ge=0.75, le=1.0)
    attack_knowledge_version: str = Field(min_length=3)
    allow_ambiguous_root_cause: Literal[True]
    preserve_pre_adjudication_annotations: Literal[True]

    @model_validator(mode="after")
    def require_frozen_attack_version(self) -> "GoldAnnotationConfig":
        if "change_me" in self.attack_knowledge_version.lower():
            raise ValueError("attack_knowledge_version must be frozen")
        return self


class AgentsConfig(Section):
    workflow: Literal["modality_routed_multi_agent"]
    roles: tuple[
        Literal["orchestrator"],
        Literal["network_investigator"],
        Literal["host_investigator"],
        Literal["correlation_agent"],
        Literal["evidence_verifier"],
    ]
    initial_window_before_seconds: int = Field(gt=0, le=60)
    initial_window_after_seconds: int = Field(gt=0, le=120)
    allow_one_window_expansion: Literal[True]
    expansion_seconds_each_side: int = Field(gt=0, le=60)
    max_total_tool_calls: int = Field(gt=0, le=24)
    max_tool_calls_per_role: int = Field(gt=0, le=8)
    max_output_tokens: int = Field(gt=0, le=3000)
    temperature: Literal[0.0]
    require_evidence_ids: Literal[True]
    allow_shell: Literal[False]
    allow_write_tools: Literal[False]
    allow_response_execution: Literal[False]
    untrusted_evidence_policy: Literal["treat_as_data_never_instructions"]
    output_schema: str = Field(min_length=3)


class LLMProviderConfig(Section):
    provider: str = Field(min_length=2)
    model: str = Field(min_length=2)
    api_key_env: str = Field(min_length=2)
    base_url_env: str | None = None
    response_format: Literal["json_schema", "json_object"] = "json_schema"
    cost_rmb_per_million_input_tokens: int | float | None = Field(default=None, ge=0.0)
    cost_rmb_per_million_output_tokens: int | float | None = Field(default=None, ge=0.0)


class LLMConfig(Section):
    primary: LLMProviderConfig
    candidates: list[LLMProviderConfig] = Field(min_length=1)
    development_selection_cases: int = Field(gt=0, le=15)
    selection_metric: Literal["grounded_evidence_f1_per_rmb"]
    total_budget_rmb: int | float = Field(gt=0.0)
    stop_noncore_at_rmb: int | float = Field(gt=0.0)
    cache_enabled: bool
    retry_provider_errors: int = Field(ge=0, le=5)
    retry_content_for_best_result: Literal[False]

    @model_validator(mode="after")
    def stop_budget_does_not_exceed_total(self) -> "LLMConfig":
        if self.stop_noncore_at_rmb > self.total_budget_rmb:
            raise ValueError("llm.stop_noncore_at_rmb exceeds total_budget_rmb")
        return self


class LoggingConfig(Section):
    redact_api_keys: Literal[True]
    redact_original_paths_in_agent_logs: Literal[True]


class ResourcesConfig(Section):
    gpu_ids: list[int] = Field(min_length=1)
    gpu_memory_gib: int = Field(gt=0)
    system_ram_gib: int = Field(gt=0)
    enforced_peak_ram_gib: int = Field(gt=0, le=16)
    disk_free_required_gib: int = Field(gt=0)


class TraceAnchorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[1]
    project: ProjectConfig
    runtime: RuntimeConfig
    paths: PathsConfig
    ingestion: IngestionConfig
    privacy_and_blinding: PrivacyConfig
    splits: SplitsConfig
    labels: LabelsConfig
    windowing: WindowingConfig
    features: FeaturesConfig
    detector: DetectorConfig
    alerting: AlertingConfig
    evidence_store: EvidenceStoreConfig
    gold_annotation: GoldAnnotationConfig
    agents: AgentsConfig
    llm: LLMConfig
    resources: ResourcesConfig
    logging: LoggingConfig

    def resolved_dict(self, config_path: Path) -> dict[str, Any]:
        # Optional typed fields that were not present in the frozen YAML must not
        # silently change the canonical configuration hash.
        data = self.model_dump(mode="json", exclude_unset=True)
        root = config_path.resolve().parent
        configured_root = self.project.project_root.resolve()
        if configured_root != root:
            raise ValueError(
                f"project.project_root resolves to {configured_root}, config is under {root}"
            )
        paths = data["paths"]
        for key, raw_value in paths.items():
            if not isinstance(raw_value, str):
                continue
            value = Path(raw_value)
            paths[key] = str(value if value.is_absolute() else configured_root / value)
        data["project"]["project_root"] = str(configured_root)
        return data


def load_config(path: str | Path) -> TraceAnchorConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return TraceAnchorConfig.model_validate(raw)


def config_hash(config: TraceAnchorConfig, config_path: str | Path) -> str:
    resolved = config.resolved_dict(Path(config_path))
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "TraceAnchorConfig",
    "ValidationError",
    "config_hash",
    "load_config",
]
