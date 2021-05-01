# coding=utf-8
from dataclasses import dataclass, field
from typing import Callable, Union

from experiment_runner.experiment_spec import PpoRudderExperimentSpec, RudderLstmExperimentSpec


@dataclass
class RudderLstmParameterSearchMap(RudderLstmExperimentSpec):
    """
    Use the `spec_instance.randomnized_spec()` methode to create a new configuration based on your specification

    Usage example:
    >>> parameter_search_spec_example = RudderLstmParameterSearchMap(
    >>>     env_name="CartPole-v1",
    >>>     env_batch_size=8,
    >>>     model_hidden_size=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    >>>     env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    >>>     seed=42,
    >>>     )

    See `experiment_runner/rudder_parameter_search_TEMPLATE.py` for a complete example
    """
    env_batch_size: Union[None, int, Callable] = field(default=None)
    model_hidden_size: Union[None, int, Callable] = field(default=None)
    env_n_trajectories: Union[None, int, Callable] = field(default=None)
    env_perct_optimal: Union[float, Callable] = field(default=0.5)
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


@dataclass
class PpoRudderParameterSearchMap(PpoRudderExperimentSpec):
    """
    Use the `spec_instance.randomnized_spec()` methode to create a new configuration based on your specification

    Usage example:
    >>> parameter_search_spec_example = PpoRudderParameterSearchMap(
    >>>     env_name="CartPole-v1",
    >>>     env_batch_size=8,
    >>>     hidden_dim=lambda: random.choice([10, 15, 20, 25, 30, 40, 60]),
    >>>     env_n_trajectories=lambda: random.choice([3000, 4000, 5000, ]),
    >>>     seed=42,
    >>>     )

    See `experiment_runner/rudder_parameter_search_TEMPLATE.py` for a complete example
    """
    steps_by_epoch: Union[None, int, Callable] = field(default=None)
    n_epoches: Union[None, int, Callable] = field(default=None)
    hidden_dim: Union[None, int, Callable] = field(default=None)
    n_hidden_layers: Union[None, int, Callable] = field(default=None)
    rudder_hidden_size: Union[None, int, Callable] = field(default=None)
    n_trajectory_per_policy: Union[None, int, Callable] = field(default=None)
    reward_delayed: bool = field(default=True)
    rew_factor: Union[float, Callable] = field(default=1.0)
    optimizer_weight_decay: Union[float, Callable] = field(default=0.0)
    optimizer_lr: Union[float, Callable] = field(default=1e-3)
    env_batch_size: Union[None, int, Callable] = field(default=None)
    env_n_trajectories: Union[None, int, Callable] = field(default=None)
    env_perct_optimal: Union[float, Callable] = field(default=0.5)
    paramSearchCallables: PpoRudderExperimentSpec = field(init=False)

    def __post_init__(self) -> None:
        """ Pull every callable argument in a shadow dataclass for later execution by .randomnized_spec(...) """
        self.paramSearchCallables = PpoRudderExperimentSpec()

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
