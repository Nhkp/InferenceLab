"""Versioned public datasets and deterministic, pre-tokenized request plans."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, StrictBool, model_validator

from inference_lab.config.io import load_experiment
from inference_lab.config.schemas import (
    ContractModel,
    Count,
    ExperimentConfig,
    GenerationConfig,
    Identifier,
    ModelConfig,
    PositiveCount,
    Sha256,
    Text,
)


class Message(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(max_length=100_000)]


class DatasetRow(ContractModel):
    dataset_name: Identifier
    dataset_version: Text
    dataset_request_id: Identifier
    synthetic: StrictBool
    target_input_tokens: PositiveCount
    target_output_tokens: PositiveCount
    messages: Annotated[list[Message], Field(min_length=1, max_length=16)]


class PlannedRequest(ContractModel):
    dataset_request_id: Identifier
    input_tokens: PositiveCount
    payload: dict[str, Any]


class RequestPlan(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    synthetic: StrictBool
    dataset_name: Identifier
    dataset_version: Text
    dataset_sha256: Sha256
    model: ModelConfig
    generation: GenerationConfig
    seed: Count
    chat_template_sha256: Sha256
    template_parameters: dict[str, bool]
    warmup: list[PlannedRequest]
    requests: Annotated[list[PlannedRequest], Field(min_length=1)]

    @model_validator(mode="after")
    def fixed_payloads(self) -> "RequestPlan":
        if self.template_parameters != {"add_generation_prompt": True}:
            raise ValueError("Plan must use the explicit generation chat template")
        if self.synthetic and (
            self.model.repo_id != "mock-model" or self.model.revision != "synthetic"
        ):
            raise ValueError("Synthetic plans require the explicit mock model identity")
        for request in self.warmup + self.requests:
            expected = self.generation.model_dump() | {
                "model": self.model.repo_id,
                "seed": self.seed,
                "stream_options": {"include_usage": True},
            }
            if set(request.payload) != set(expected) | {"messages"}:
                raise ValueError("Plan payload fields differ from the canonical generation subset")
            if any(request.payload[key] != value for key, value in expected.items()):
                raise ValueError("Plan payload changed generation/model options")
            GenerationConfig.model_validate(
                {key: request.payload[key] for key in GenerationConfig.model_fields}
            )
            if type(request.payload["seed"]) is not int:
                raise ValueError("Request seed must be an integer")
            messages = request.payload["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("Plan messages must be a nonempty list")
            for message in messages:
                Message.model_validate(message)
        return self


def encoded(value: ContractModel) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def template_digest(template: str) -> str:
    """Match 00B's UTF-8 JSON-string convention, rather than hashing raw text."""
    return digest(json.dumps(template, ensure_ascii=False, sort_keys=True).encode())


def dataset(path: Path) -> tuple[bytes, list[DatasetRow]]:
    data = path.read_bytes()
    rows = [DatasetRow.model_validate_json(line) for line in data.splitlines() if line.strip()]
    if not rows or len({row.dataset_request_id for row in rows}) != len(rows):
        raise ValueError("Dataset needs nonempty rows and unique request IDs")
    if (
        len(
            {
                (
                    r.dataset_name,
                    r.dataset_version,
                    r.synthetic,
                    r.target_input_tokens,
                    r.target_output_tokens,
                )
                for r in rows
            }
        )
        != 1
    ):
        raise ValueError("One dataset must describe one versioned workload and shape")
    return data, rows


def prepare_plan(
    config: ExperimentConfig,
    path: Path,
    *,
    tokenize: Callable[[list[dict[str, str]]], int],
    template: str,
) -> RequestPlan:
    data, rows = dataset(path)
    first = rows[0]
    prepared = []
    for row in rows:
        messages = [message.model_dump() for message in row.messages]
        count = tokenize(messages)
        config.validate_input_length(count)
        if (
            count != row.target_input_tokens
            or config.generation.max_tokens != row.target_output_tokens
        ):
            raise ValueError(
                "Dataset input/output shape does not match templated tokens and generation"
            )
        prepared.append(
            PlannedRequest(
                dataset_request_id=row.dataset_request_id,
                input_tokens=count,
                payload=config.generation.model_dump()
                | {
                    "model": config.model.repo_id,
                    "messages": messages,
                    "seed": config.workload.seed,
                    "stream_options": {"include_usage": True},
                },
            )
        )
    randomizer = random.Random(config.workload.seed)
    return RequestPlan(
        synthetic=first.synthetic,
        dataset_name=first.dataset_name,
        dataset_version=first.dataset_version,
        dataset_sha256=digest(data),
        model=config.model,
        generation=config.generation,
        seed=config.workload.seed,
        chat_template_sha256=template_digest(template),
        template_parameters={"add_generation_prompt": True},
        warmup=randomizer.choices(prepared, k=config.warmup.requests_per_point),
        requests=randomizer.choices(prepared, k=config.measurement.requests_per_run),
    )


def validate_plan(plan: RequestPlan, config: ExperimentConfig, path: Path) -> None:
    RequestPlan.model_validate(plan.model_dump())
    data, rows = dataset(path)
    if (
        plan.dataset_sha256 != digest(data)
        or plan.model != config.model
        or plan.generation != config.generation
        or plan.seed != config.workload.seed
        or len(plan.requests) != config.measurement.requests_per_run
        or len(plan.warmup) != config.warmup.requests_per_point
        or plan.synthetic != rows[0].synthetic
        or plan.dataset_name != rows[0].dataset_name
        or plan.dataset_version != rows[0].dataset_version
    ):
        raise ValueError("Prepared plan does not match this dataset/experiment")
    by_id = {row.dataset_request_id: row for row in rows}
    for request in plan.requests + plan.warmup:
        row = by_id.get(request.dataset_request_id)
        if row is None or request.payload["messages"] != [
            message.model_dump() for message in row.messages
        ]:
            raise ValueError("Planned messages differ from the versioned dataset")
        if request.input_tokens != row.target_input_tokens:
            raise ValueError("Planned token length differs from the dataset shape")
        config.validate_input_length(request.input_tokens)


def tokenizer(config: ExperimentConfig) -> Any:
    config.model.require_immutable()
    for package, version in (
        ("transformers", "4.55.2"),
        ("tokenizers", "0.21.4"),
        ("jinja2", "3.1.6"),
    ):
        if importlib.metadata.version(package) != version:
            raise ValueError(
                "Prepare with qualified tokenizer dependencies; runner uses the saved plan"
            )
    module = importlib.import_module("transformers")
    return module.AutoTokenizer.from_pretrained(
        config.model.tokenizer_repo_id,
        revision=config.model.tokenizer_revision,
        trust_remote_code=False,
    )


def reference_datasets(output: Path, tok: Any) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, input_count, output_count in (
        ("short-chat-v1", 128, 128),
        ("long-context-512-v1", 512, 128),
        ("long-context-2048-v1", 2048, 128),
        ("generation-heavy-v1", 128, 512),
    ):
        rows = []
        for index in range(4):
            content = (
                f"Synthetic workload {index}. Continue counting integers starting at one. Padding:"
            )
            messages = [{"role": "user", "content": content}]
            count = len(
                tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            )
            # ponytail: verified one-token ' x' padding for this pinned tokenizer;
            # reject another tokenizer rather than silently approximating target lengths.
            messages[0]["content"] += " x" * (input_count - count)
            if (
                len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                != input_count
            ):
                raise ValueError("Tokenizer cannot represent the requested exact padding shape")
            rows.append(
                DatasetRow(
                    dataset_name=name,
                    dataset_version="1",
                    dataset_request_id=f"{name}-{index:03d}",
                    synthetic=False,
                    target_input_tokens=input_count,
                    target_output_tokens=output_count,
                    messages=[Message.model_validate(message) for message in messages],
                )
            )
        with (output / f"{name}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(row.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--synthetic", action="store_true", help="Fixed mock counts, never real tokenization"
    )
    parser.add_argument(
        "--generate-datasets",
        action="store_true",
        help="Write reference JSONL files instead of a plan",
    )
    args = parser.parse_args()
    config = load_experiment(args.experiment)
    if args.synthetic:
        if args.generate_datasets or not dataset(Path(config.workload.dataset))[1][0].synthetic:
            parser.error("Synthetic preparation requires a synthetic mock dataset")
        # ponytail: the mock has fixed synthetic input=3, not a tokenizer. Real datasets
        # use the qualified tokenizer and are never accepted through this fixture path.
        plan = prepare_plan(
            config,
            Path(config.workload.dataset),
            tokenize=lambda _: 3,
            template="synthetic/mock-v1",
        )
    else:
        tok = tokenizer(config)
        if args.generate_datasets:
            reference_datasets(args.output, tok)
            provenance = {
                "scope": "CPU tokenization only; no model generation or GPU benchmark",
                "tokenizer_repo_id": config.model.tokenizer_repo_id,
                "tokenizer_revision": config.model.tokenizer_revision,
                "chat_template_sha256": template_digest(tok.chat_template),
                "chat_template_hash_encoding": (
                    "UTF-8 JSON string, ensure_ascii=false, sort_keys=true"
                ),
                "template_parameters": {"add_generation_prompt": True},
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("transformers", "tokenizers", "jinja2")
                },
                "datasets": {
                    path.name: {
                        "sha256": digest(path.read_bytes()),
                        "rows": len(dataset(path)[1]),
                        "input_tokens": dataset(path)[1][0].target_input_tokens,
                        "output_tokens": dataset(path)[1][0].target_output_tokens,
                    }
                    for path in sorted(args.output.glob("*.jsonl"))
                },
            }
            with (args.output / "tokenizer-evidence.json").open("x") as stream:
                json.dump(provenance, stream, indent=2)
                stream.write("\n")
            return
        if not isinstance(tok.chat_template, str):
            raise ValueError("Tokenizer must expose one explicit chat template")
        plan = prepare_plan(
            config,
            Path(config.workload.dataset),
            template=tok.chat_template,
            tokenize=lambda messages: len(
                tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            ),
        )
    with args.output.open("xb") as stream:
        stream.write(encoded(plan))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, yaml.YAMLError):
        raise SystemExit(
            "Plan preparation failed: check config, dataset, qualified tokenizer "
            "dependencies, and fresh output path. No model weights are needed."
        ) from None
