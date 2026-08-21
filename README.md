
# TraceWeaver

TraceWeaver is a reproducible implementation of evidence-grounded host and
network incident investigation. It turns independent network/host alerts into
read-only evidence queries and validates every final claim against Evidence IDs.

This directory is the public source bundle. It contains the runnable research
implementation, frozen public schemas, prompts, and experiment configurations.
It deliberately does not contain raw LID-DS data, annotations, model outputs,
API credentials, private evaluator databases, approval records, execution
audits, or process-test scripts.

## Requirements

- Linux or WSL2
- Python 3.10
- Conda environment named `TraceWeaver`
- PyTorch 2.1.2 with CUDA 11.8 (or a CPU build for small smoke runs)
- The Python dependencies in `requirements.txt`
- A local copy of the LID-DS data; raw files are never modified

Install the package from this directory:

```bash
conda activate TraceAnchor
python -m pip install -e '.[dev]'
```

## Configuration and data

Create a working configuration and edit the two data paths:

```bash
cp project.example.yml project.yml
# Set paths.raw_data_root and paths.example_scenario in project.yml.
cp configs/traceweaver.env.example /secure/path/traceweaver.env
chmod 600 /secure/path/traceweaver.env
```

The environment file stores variable names only. Never commit credentials or
provider responses. Provider-backed experiments require the corresponding
runtime variables; deterministic development runs can use `--provider mock`.

## Formal experiment flow

Run the stages in order. Each stage writes derived data below the configured
`artifacts/` paths and supports resumable output where applicable.

```bash
# Environment and configuration
python -m traceweaver.cli env-check --config project.yml

# Raw manifest, ingestion, and QA
python -m traceweaver.cli manifest --config project.yml

# Independent causal detectors
python -m traceweaver.cli train-detector --config project.yml --channel network --seed 42 --resume

# Isolated evidence ledgers and read-only tool checks
python -m traceweaver.cli validate-blind-view --config project.yml

# Evaluator-side gold workflow
python -m traceweaver.cli sample-gold --config project.yml --split development

## Public package layout

```text
src/traceweaver/
  ingest/       LID-DS manifesting, parsing, and validation
  build/        scene splits and causal feature construction
  detector/     independent host/network causal TCNs and alert calibration
  evidence/     blind ledger, typed read-only tools, and isolation checks
  agents/       provider adapters, broker, workflow, verifier, and schemas
  annotation/   evaluator-side gold annotation workflow
configs/        environment template and frozen experiment settings
prompts/        agent and baseline prompts
schemas/        public output and manifest schemas
```

## Reproducibility and scope

The public bundle is sufficient to rebuild the data-processing, detector,
evidence, Development, baseline, ablation-definition, and provider-free
adversarial stages when the user supplies the permitted data and runtime
configuration. Historical artifacts are intentionally not copied into this
directory so that generated results cannot be mistaken for a fresh run.
Commands that depend on a prior completion marker, gold annotation, or paid
provider approval fail closed until that prerequisite is created in the local
run; this is expected and prevents accidental protocol bypass.

Agent Test paid execution, recovery, authorization, and monitoring code is
excluded because it is process-control code rather than part of the public
scientific implementation. Test data and evaluator-only gold remain private.
Do not use Agent Test outputs to tune prompts, models, tools, schemas, budgets,
thresholds, or gold annotations.

##
## Thanks

[1] https://github.com/LID-DS/LID-DS

[2] https://link.springer.com/chapter/10.1007/978-3-031-35190-7_6