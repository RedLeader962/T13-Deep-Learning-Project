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


@dataclass(frozen=True)
class RudderLstmExperimentSpec(ExperimentSpec):
    env_batch_size: Optional[int]
    model_hidden_size: Optional[int]
    env_n_trajectories: Optional[int]
    env_perct_optimal: Optional[float]
    n_epoches: Optional[int]
    optimizer_weight_decay: Optional[float]
    optimizer_lr: Optional[float]
    seed: Optional[int]
