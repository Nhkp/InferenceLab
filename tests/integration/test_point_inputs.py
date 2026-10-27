import json
from pathlib import Path

import pytest
from test_streaming_client import DONE, USAGE, event, server
from test_workload import ROOT, configuration, planned, scheduler

from inference_lab.benchmark.workload import run_point
from inference_lab.config.schemas import ExperimentConfig, ExperimentRun

pytestmark = pytest.mark.integration


async def test_point_uses_frozen_dataset_policy_and_exports_drain(tmp_path: Path) -> None:
    async def source():
        yield event("synthetic", "length") + USAGE + DONE

    async with server(source, metrics=lambda: scheduler()) as url:
        config = configuration(url, count=1, warmup=0, concurrency=1)
        plan = planned(config)
        dataset = tmp_path / "dataset.jsonl"
        dataset.write_bytes((ROOT / config.workload.dataset).read_bytes())
        data = config.model_dump(mode="json")
        data["workload"]["dataset"] = "missing-original-dataset.jsonl"
        data["policy_file"] = "missing-original-policy.yaml"
        config = ExperimentConfig.model_validate(data)
        policy = json.loads((ROOT / "tests/data/synthetic-policy.json").read_text())
        policy.update(workload_id="mock-v1", min_repetitions=1)
        snapshot = tmp_path / "policy.json"
        snapshot.write_text(json.dumps(policy))
        run = ExperimentRun(
            campaign_id="frozen",
            run_id="frozen",
            point=0,
            repetition=0,
            concurrency=1,
            config=config,
        )
        result = await run_point(
            run, plan, tmp_path / "run", dataset_path=dataset, policy_path=snapshot
        )
    assert result.complete and result.workload.conformant
    assert result.config.workload.dataset == "missing-original-dataset.jsonl"
    assert (tmp_path / "run/dataset.jsonl").read_bytes() == dataset.read_bytes()
    drain = json.loads((tmp_path / "run/drain.json").read_text())
    assert all(
        drain[key] == "VALIDATED" for key in ("before_warmup", "after_warmup", "after_measurement")
    )
    assert (tmp_path / "run/policy.json").is_file()
