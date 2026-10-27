import json
from copy import deepcopy
from pathlib import Path

import pytest

from inference_lab.benchmark.campaign import pair_protocol, paired_plan_hash, schedule
from inference_lab.benchmark.plan import prepare_plan
from inference_lab.config.io import load_experiment
from inference_lab.config.schemas import ExperimentConfig, RegressionPolicy

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def protocol(*, real=False):
    config = load_experiment(
        ROOT / ("experiments/qwen3b-fp16.yaml" if real else "experiments/mock-stream.yaml")
    )
    data = config.model_dump(mode="json")
    data["workload"]["concurrency"] = [1, 2]
    data["measurement"]["repetitions"] = 3
    config = ExperimentConfig.model_validate(data)
    plan = prepare_plan(
        config,
        ROOT / config.workload.dataset,
        tokenize=lambda _: 128 if real else 3,
        template="CPU protocol fixture",
    )
    policy_data = json.loads((ROOT / "tests/data/synthetic-policy.json").read_text())
    policy_data["workload_id"] = plan.dataset_name
    policy_data["comparison"]["allowed_changes"] = []
    policy = RegressionPolicy.model_validate(policy_data)
    return (
        {"baseline": config, "candidate": deepcopy(config)},
        {"baseline": plan, "candidate": deepcopy(plan)},
        {"baseline": policy, "candidate": deepcopy(policy)},
    )


def test_schedule_balances_pair_order_and_has_unique_global_ids() -> None:
    configs, plans, policies = protocol()
    order = schedule("campaign-one", configs)
    assert [(e.condition, e.repetition) for e in order[:6]] == [
        ("baseline", 0),
        ("candidate", 0),
        ("candidate", 1),
        ("baseline", 1),
        ("baseline", 2),
        ("candidate", 2),
    ]
    assert len(order) == 12 and len({e.run_id for e in order}) == 12
    assert not {e.run_id for e in order} & {e.run_id for e in schedule("campaign-two", configs)}
    comparison, differences = pair_protocol(configs, plans, policies)
    assert comparison.kind == "regression" and differences == []
    assert paired_plan_hash(plans["baseline"]) == paired_plan_hash(plans["candidate"])


@pytest.mark.parametrize(
    "change", ["undeclared", "points", "repetitions", "plan", "policy", "missing_policy"]
)
def test_pairing_rejects_incompatible_inputs(change) -> None:
    configs, plans, policies = protocol()
    candidate = configs["candidate"].model_dump(mode="json")
    if change == "undeclared":
        candidate["runtime"]["gpu_memory_utilization"] = 0.84
    elif change == "points":
        candidate["workload"]["concurrency"] = [1, 4]
    elif change == "repetitions":
        candidate["measurement"]["repetitions"] = 4
    elif change == "plan":
        plans["candidate"].requests.reverse()
        plans["candidate"].chat_template_sha256 = "b" * 64
    elif change == "policy":
        policies["candidate"].policy_id = "different-policy"
    else:
        policies["candidate"] = None
    configs["candidate"] = ExperimentConfig.model_validate(candidate)
    with pytest.raises(ValueError):
        pair_protocol(configs, plans, policies)


def test_declared_config_change_is_recorded_but_not_a_verdict() -> None:
    configs, plans, policies = protocol()
    configs["candidate"].runtime.gpu_memory_utilization = 0.84
    for policy in policies.values():
        policy.comparison.allowed_changes = ["runtime.gpu_memory_utilization"]
    comparison, differences = pair_protocol(configs, plans, policies)
    assert differences[0].path == "runtime.gpu_memory_utilization"
    assert differences[0].baseline == 0.85 and differences[0].candidate == 0.84
    assert comparison.kind == "regression"


def test_real_regression_pair_requires_preexisting_calibration_and_three_repetitions() -> None:
    configs, plans, policies = protocol(real=True)
    with pytest.raises(ValueError, match="calibration"):
        pair_protocol(configs, plans, policies)
    for policy in policies.values():
        policy.calibration_ref = "calibration/evidence.json"
    comparison, _ = pair_protocol(configs, plans, policies)
    assert comparison.kind == "regression"
    for config in configs.values():
        config.measurement.repetitions = 2
    with pytest.raises(ValueError, match="fewer repetitions"):
        pair_protocol(configs, plans, policies)


def test_tp_requires_exploratory_same_checkpoint_and_explicit_count_change() -> None:
    configs, plans, policies = protocol(real=True)
    configs["candidate"].runtime.tensor_parallel_size = 2
    for policy in policies.values():
        policy.comparison.allowed_changes = ["runtime.tensor_parallel_size"]
    with pytest.raises(ValueError, match="exploratory"):
        pair_protocol(configs, plans, policies)
    for policy in policies.values():
        policy.comparison.kind = "exploratory"
    comparison, differences = pair_protocol(configs, plans, policies)
    assert (
        comparison.kind == "exploratory" and differences[0].path == "runtime.tensor_parallel_size"
    )


def test_awq_requires_equal_resources_and_is_not_a_standard_gate() -> None:
    configs, plans, policies = protocol(real=True)
    configs["candidate"].runtime.quantization = "awq"
    for policy in policies.values():
        policy.comparison.allowed_changes = ["runtime.quantization"]
    with pytest.raises(ValueError, match="exploratory"):
        pair_protocol(configs, plans, policies)
    for policy in policies.values():
        policy.comparison.kind = "exploratory"
    assert pair_protocol(configs, plans, policies)[0].kind == "exploratory"
    configs["candidate"].runtime.tensor_parallel_size = 2
    for policy in policies.values():
        policy.comparison.allowed_changes.append("runtime.tensor_parallel_size")
    with pytest.raises(ValueError, match="same TP"):
        pair_protocol(configs, plans, policies)
