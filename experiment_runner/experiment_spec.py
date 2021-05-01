# coding=utf-8
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExperimentSpec:
    show_plot: bool = field(default=True)
    print_to_consol: bool = field(default=True)
    spec_name: Optional[str] = field(default=None)
    spec_id: Optional[int] = field(default=None)
    env_name: Optional[str] = field(default=None)
    comment: Optional[str] = field(default=None)
    results: Optional[Any] = field(init=False, default=None)
    seed: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        indent = "   "
        spec_str = f"{indent}{self.__class__.__name__} {'›››'}"
        if self.spec_name:
            spec_str += "\n{}{:>26} : {}".format(indent, 'spec_name', self.spec_name)

        for each_key, each_value in self.__dict__.items():
            if each_key != 'spec_name':
                if each_value is not None:
                    spec_str += ("\n{}{:>26} : {}".format(indent, each_key, each_value))

        spec_str += f"\n\n{indent}{''}\n"
        return spec_str


@dataclass
class PpoExperimentSpec(ExperimentSpec):
    steps_by_epoch: Optional[int] = field(default=None)
    n_epoches: Optional[int] = field(default=None)
    hidden_dim: Optional[int] = field(default=None)
    n_hidden_layers: Optional[int] = field(default=None)
    n_trajectory_per_policy: Optional[int] = field(default=None)
    reward_delayed: bool = field(default=True)
    rew_factor: float = field(default=1.0)
    optimizer_weight_decay: float = field(default=0.0)
    optimizer_lr: float = field(default=1e-3)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class PpoRudderExperimentSpec(ExperimentSpec):
    steps_by_epoch: Optional[int] = field(default=None)
    n_epoches: Optional[int] = field(default=None)
    hidden_dim: Optional[int] = field(default=None)
    n_hidden_layers: Optional[int] = field(default=None)
    rudder_hidden_size: Optional[int] = field(default=None)
    n_trajectory_per_policy: Optional[int] = field(default=None)
    reward_delayed: bool = field(default=True)
    rew_factor: float = field(default=1.0)
    optimizer_weight_decay: float = field(default=0.0)
    optimizer_lr: float = field(default=1e-3)
    env_batch_size: Optional[int] = field(default=None)
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: float = field(default=0.5)


    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderExperimentSpec(ExperimentSpec):
    n_epoches: Optional[int] = field(default=None)
    env_batch_size: Optional[int] = field(default=None)
    loader_batch_size: Optional[int] = field(default=None)


    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderLstmExperimentSpec(ExperimentSpec):
    env_batch_size: Optional[int] = field(default=None)
    model_hidden_size: Optional[int] = field(default=None)
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: float = field(default=0.5)
    rew_factor: float = field(default=1.0)
    n_epoches: Optional[int] = field(default=None)
    optimizer_weight_decay: float = field(default=0.0)
    optimizer_lr: float = field(default=1e-3)


    def __repr__(self) -> str:
        return super().__repr__()


