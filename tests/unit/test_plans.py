import json
from copy import deepcopy
from pathlib import Path

import pytest

from inference_lab.benchmark.plan import (
    RequestPlan,
    dataset,
    digest,
    encoded,
    prepare_plan,
    validate_plan,
)
from inference_lab.config.io import load_experiment

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_plan_is_deterministic_serializable_and_shape_checked() -> None:
    config = load_experiment(ROOT / "experiments/mock-stream.yaml")
    path = ROOT / config.workload.dataset
    plan = prepare_plan(config, path, tokenize=lambda _: 3, template="synthetic/mock-v1")
    assert encoded(plan) == encoded(
        prepare_plan(config, path, tokenize=lambda _: 3, template="synthetic/mock-v1")
    )
    assert RequestPlan.model_validate_json(encoded(plan)) == plan
    assert plan.synthetic and len(plan.requests) == 8 and len(plan.warmup) == 4
    validate_plan(plan, config, path)
    assert plan.dataset_sha256 == digest(path.read_bytes())
    with pytest.raises(ValueError, match="shape"):
        prepare_plan(config, path, tokenize=lambda _: 4, template="synthetic/mock-v1")
    changed = deepcopy(plan)
    changed.requests[0].payload["seed"] = 43
    with pytest.raises(ValueError):
        RequestPlan.model_validate(changed.model_dump())
    changed = deepcopy(plan)
    changed.requests[0].payload["messages"][0]["content"] = "tampered"
    with pytest.raises(ValueError, match="messages differ"):
        validate_plan(changed, config, path)


def test_dataset_rejects_duplicate_ids_and_mixed_versions(tmp_path: Path) -> None:
    line = (ROOT / "experiments/datasets/mock-v1.jsonl").read_text()
    path = tmp_path / "data.jsonl"
    path.write_text(line + line)
    with pytest.raises(ValueError, match="unique"):
        dataset(path)
    row = json.loads(line)
    row.update(dataset_request_id="another", dataset_version="different")
    path.write_text(line + json.dumps(row))
    with pytest.raises(ValueError, match="one versioned"):
        dataset(path)


def test_reference_datasets_have_preserved_tokenization_provenance() -> None:
    metadata = json.loads((ROOT / "experiments/datasets/tokenizer-evidence.json").read_text())
    pin = json.loads((ROOT / "qualification/kaggle-t4-20261025-c/model-pin.json").read_text())
    assert metadata["chat_template_sha256"] == pin["chat_template_sha256"]
    assert metadata["tokenizer_revision"] == pin["tokenizer_revision"]
    assert metadata["versions"] == {
        "transformers": "4.55.2",
        "tokenizers": "0.21.4",
        "jinja2": "3.1.6",
    }
    assert {(d["input_tokens"], d["output_tokens"]) for d in metadata["datasets"].values()} == {
        (128, 128),
        (512, 128),
        (2048, 128),
        (128, 512),
    }
    for filename, facts in metadata["datasets"].items():
        data, rows = dataset(ROOT / "experiments/datasets" / filename)
        assert digest(data) == facts["sha256"]
        assert len(rows) == facts["rows"]
        assert all(not row.synthetic for row in rows)


@pytest.mark.parametrize("change", [{"n": True}, {"seed": True}, {"stream": "true"}])
def test_prepared_payloads_cannot_bypass_strict_generation_types(change) -> None:
    config = load_experiment(ROOT / "experiments/mock-stream.yaml")
    plan = prepare_plan(
        config, ROOT / config.workload.dataset, tokenize=lambda _: 3, template="synthetic/mock-v1"
    )
    data = plan.model_dump()
    data["requests"][0]["payload"].update(change)
    with pytest.raises(ValueError):
        RequestPlan.model_validate(data)


def test_dataset_generation_never_mixes_with_existing_files(tmp_path: Path) -> None:
    from inference_lab.benchmark.plan import reference_datasets

    with pytest.raises(FileExistsError):
        reference_datasets(tmp_path, None)
