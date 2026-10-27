# Repository Guidelines

## Project Structure & Module Organization

InferenceLab is a Python 3.12 bootstrap. Application code lives in `src/inference_lab/`; CPU tests live in `tests/{unit,integration}`. The gateway currently exposes only `/health`; `benchmark/mock_server.py` provides a separate synthetic SSE upstream.

`inference_lab_codex_prompts_v5_1/` contains design documents and implementation prompts, not working features. Start with `START_HERE.md`; consult `PROJECT_DESIGN_DOCUMENT.md` and `BENCHMARK_CONTRACT.md` before implementing a stage. Follow the pack’s README execution order. Optional proposals require explicit selection.

`config` contains canonical schemas, YAML loading, and settings; see `docs/architecture.md`. `serving` contains the external vLLM launch/probe CLI and HTTP streaming client; see `docs/serving.md`. `benchmark` contains deterministic plan preparation, one-point closed-loop measurement, and raw-data metric replay; see `docs/benchmarking.md`. `registry`, `telemetry`, and `storage` are reserved namespaces. Qualification evidence lives in `qualification/`. CPU containers use `compose.yaml`; `compose.gpu.yaml` is a separate opt-in Linux GPU profile. Normative links in `docs/` reuse the implementation pack's canonical documents.

## Build, Test, and Development Commands

- `make install`: install locked CPU dependencies and development tools with uv.
- `make lint typecheck test`: check Ruff formatting/lint, strict mypy, and CPU tests.
- `make format`: apply Ruff fixes and formatting to source and tests.
- `make run` / `make mock`: start the health API / synthetic upstream.
- `make up` / `make down`: start / stop the CPU Compose services.
- `make build`: build the Python package.

`.python-version` pins qualified Python 3.12.12. GPU dependencies are an optional extra; default installation must stay CPU-only. Update `uv.lock` when dependencies change. Avoid reformatting preserved qualification artifacts.

## Coding Style & Naming Conventions

Use four-space Python indentation, type annotations, `snake_case` functions/modules, and `PascalCase` classes. Reuse existing code, standard-library features, and installed dependencies before adding code or packages. Avoid speculative abstractions.

Trace the complete flow and all callers before fixing shared behavior. Prefer the smallest correct change. Mark intentional shortcuts with `ponytail:` comments; name any known ceiling and upgrade path.

## Testing Guidelines

Leave one runnable check for nontrivial logic. Use `test_*.py` files and the `unit`, `integration`, `gpu`, and `slow` markers. `make test` excludes GPU/slow checks. No pytest GPU tests exist yet; empty collection is not validation. CPU tests must not require CUDA or download model weights. No coverage threshold exists. Report skipped checks separately from passes; see `docs/testing.md`.

## Commit & Pull Request Guidelines

Use Conventional Commits, such as `feat(benchmark): add request measurements`. Keep independently reversible changes in dependency order, with tests beside their feature. Keep commits focused. PRs should explain behavior, validation performed, limitations, and relevant issues. Include artifacts for performance claims; never present unexecuted GPU qualification as verified.

## Security & Configuration

Keep secrets in environment variables and examples free of credentials. Never log authorization headers, secrets, or complete prompts and responses.


## InferenceLab project instructions

Before any task, read:
- docs/PROJECT_DESIGN_DOCUMENT.md
- docs/BENCHMARK_CONTRACT.md

Never:
- invent benchmark metrics;
- silently change dtype/backend/model;
- treat a skipped GPU test as validated;
- compare incompatible benchmark runs.

The reference serving hardware is Kaggle with NVIDIA T4.

JSON benchmark artifacts are authoritative.
PostgreSQL is secondary/indexing storage.

Client-side TPOT must be named:
client_tpot_estimate_ms

Do not call it runtime TPOT.

All performance claims require measured evidence.
