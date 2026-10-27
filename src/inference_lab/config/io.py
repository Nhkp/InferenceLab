"""Safe canonical YAML loading. Paths in YAML are relative to the supplied project root."""

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode

from inference_lab.config.schemas import ExperimentConfig, RegressionPolicy


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        mapping: dict[Hashable, Any] = {}
        for key, value in self.construct_pairs(node, deep=deep):
            if not isinstance(key, str) or key in mapping:
                raise ValueError("YAML mappings require unique string keys")
            mapping[key] = value
        return mapping


def _read_yaml(path: Path) -> object:
    with path.open() as stream:
        return yaml.load(stream, Loader=_UniqueKeyLoader)


def load_policy(path: Path) -> RegressionPolicy:
    return RegressionPolicy.model_validate(_read_yaml(path))


def load_experiment(
    path: Path, *, published: bool = False, project_root: Path = Path(".")
) -> ExperimentConfig:
    experiment = ExperimentConfig.model_validate(_read_yaml(path), context={"published": published})
    if experiment.policy_file is not None:
        policy = load_policy(project_root / experiment.policy_file)
        policy.validate_experiment(experiment, published=published)
    return experiment
