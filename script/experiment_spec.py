# coding=utf-8
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentSpec:
    show_plot: Optional[bool]


@dataclass(frozen=True)
class PpoExperimentSpec(ExperimentSpec):
    steps_by_epoch: Optional[int]
    n_epoches: Optional[int]
    hidden_dim: Optional[int]
    n_hidden_layers: Optional[int]
    n_trajectory_per_policy: Optional[int]


@dataclass(frozen=True)
class RudderExperimentSpec(ExperimentSpec):
    n_epoches: Optional[int]
    env_batch_size: Optional[int]
    loader_batch_size: Optional[int]