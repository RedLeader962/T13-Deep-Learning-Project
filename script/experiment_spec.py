# coding=utf-8
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union


@dataclass
class ExperimentSpec:
    show_plot: Optional[bool] = field(default=None)
    name: Optional[str] = field(default=None)
    spec_id: Optional[int] = field(default=None)
    env_name: Optional[str] = field(default=None)
    comment: Optional[str] = field(default=None)
    results: Optional[Any] = field(init=False, default=None)

    def __repr__(self) -> str:
        indent = "   "
        spec_str = f"{indent}{self.__class__.__name__} {'›››'}"
        if self.name:
            spec_str += "\n{}{:>26} : {}".format(indent, 'name', self.name)

        for each_key, each_value in self.__dict__.items():
            if each_key != 'name':
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
    seed: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderExperimentSpec(ExperimentSpec):
    n_epoches: Optional[int] = field(default=None)
    env_batch_size: Optional[int] = field(default=None)
    loader_batch_size: Optional[int] = field(default=None)
    seed: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderLstmExperimentSpec(ExperimentSpec):
    env_name: str = "CartPole-v1"
    env_batch_size: Optional[int] = field(default=None)
    model_hidden_size: Optional[int] = field(default=None)
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: Optional[float] = field(default=None)
    n_epoches: Optional[int] = field(default=None)
    optimizer_weight_decay: Optional[float] = field(default=None)
    optimizer_lr: Optional[float] = field(default=None)
    seed: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderLstmParameterSearchMap(RudderLstmExperimentSpec):
    env_batch_size: Union[None, int, Callable] = field(default=None)
    model_hidden_size: Union[None, int, Callable] = field(default=None)
    env_n_trajectories: Union[None, int, Callable] = field(default=None)
    env_perct_optimal: Union[None, float, Callable] = field(default=None)
    n_epoches: Union[None, int, Callable] = field(default=None)
    optimizer_weight_decay: Union[None, float, Callable] = field(default=None)
    optimizer_lr: Union[None, float, Callable] = field(default=None)
    paramSearchCallables: RudderLstmExperimentSpec = field(init=False)

    def __post_init__(self):
        self.paramSearchCallables = RudderLstmExperimentSpec()

        for each_key, each_value in self.__dict__.items():
            if callable(each_value):
                self.paramSearchCallables.__setattr__(each_key, each_value)
                self.__setattr__(each_key, each_value())

    def randomnized_spec(self) -> None:
        for each_key, each_value in self.paramSearchCallables.__dict__.items():
            if callable(each_value):
                self.__setattr__(each_key, each_value())

    def __repr__(self) -> str:
        return super().__repr__()
