# coding=utf-8
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union


@dataclass
class ExperimentSpec:
    show_plot: bool = field(default=True)
    print_to_consol: bool = field(default=True)
    spec_name: Optional[str] = field(default=None)
    spec_id: Optional[int] = field(default=None)
    env_name: Optional[str] = field(default=None)
    comment: Optional[str] = field(default=None)
    results: Optional[Any] = field(init=False, default=None)

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
    env_batch_size: Optional[int] = field(default=None)
    model_hidden_size: Optional[int] = field(default=None)
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: float = field(default=0.5)
    env_rew_factor: float = field(default=1.0)
    n_epoches: Optional[int] = field(default=None)
    optimizer_weight_decay: float = field(default=0.0)
    optimizer_lr: float = field(default=1e-3)
    seed: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderLstmParameterSearchMap(RudderLstmExperimentSpec):
    """
    Use the `lstm_spec_instance.randomnized_spec()` methode to create a new configuration based on your specification

    Usage example:
    >>> parameter_search_spec_example = RudderLstmParameterSearchMap(
    >>>     env_name="CartPole-v1",
    >>>     env_batch_size=8,
    >>>     model_hidden_size=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    >>>     env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    >>>     seed=42,
    >>>     )

    See `experiment_runner/parameter_search_TEMPLATE.py` for a complete example
    """
    env_batch_size: Union[None, int, Callable] = field(default=None)
    model_hidden_size: Union[None, int, Callable] = field(default=None)
    env_n_trajectories: Union[None, int, Callable] = field(default=None)
    env_perct_optimal: Union[float, Callable] = field(default=0.5)
    env_rew_factor: Union[float, Callable] = field(default=1.0)
    n_epoches: Union[None, int, Callable] = field(default=None)
    optimizer_weight_decay: Union[float, Callable] = field(default=0.0)
    optimizer_lr: Union[float, Callable] = field(default=1e-3)
    paramSearchCallables: RudderLstmExperimentSpec = field(init=False)

    def __post_init__(self) -> None:
        """ Pull every callable argument in a shadow dataclass for later execution by .randomnized_spec(...) """
        self.paramSearchCallables = RudderLstmExperimentSpec()

        for each_key, each_value in self.__dict__.items():
            if callable(each_value):
                self.paramSearchCallables.__setattr__(each_key, each_value)
                self.__setattr__(each_key, each_value())
        return None

    def randomnized_spec(self) -> None:
        """ Radomnize every field with a callable argument """
        for each_key, each_value in self.paramSearchCallables.__dict__.items():
            if callable(each_value):
                self.__setattr__(each_key, each_value())
        return None

    def __repr__(self) -> str:
        indent = "   "
        spec_str = f"{indent}{self.__class__.__name__} {'›››'}"
        if self.spec_name:
            spec_str += "\n{}{:>26} : {}".format(indent, 'spec_name', self.spec_name)

        for each_key, each_value in self.__dict__.items():
            if each_key != 'spec_name':
                if (each_value is not None) and (each_key != 'paramSearchCallables'):
                    spec_str += ("\n{}{:>26} : {}".format(indent, each_key, each_value))

        spec_str += f"\n\n{indent}{''}\n"
        return spec_str
