# coding=utf-8
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
# from experiment_runner import experiment_runner_organizer as the_organizer
from experiment_runner.constant import EXPERIMENT_RUN_DIR


@dataclass
class ExperimentSpec:
    env_name: str = field(default="CartPole-v1")
    show_plot: bool = field(default=True)
    print_to_consol: bool = field(default=True)
    spec_name: Optional[str] = field(default=None)
    comment: Optional[str] = field(default=None)
    results: Optional[Any] = field(init=False, default=None)
    seed: Optional[int] = field(default=None)
    experiment_tag: Optional[str] = field(default=None)
    experiment_dir: Optional[str] = field(default=None)
    is_batch_spec: bool = field(default=False)
    batch_tag: Optional[str] = field(default=None)
    batch_dir: Optional[str] = field(default=None)
    root_experiment_dir: str = field(default=EXPERIMENT_RUN_DIR)
    experiment_path: Optional[os.PathLike] = field(default=None)
    spec_idx: Optional[int] = field(default=None)

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

    def __post_init__(self):
        self.experiment_path = self.get_spec_run_path()

    def get_spec_run_dir(self) -> str:
        if self.experiment_path is None:

            if self.is_batch_spec:
                exp_dir = "run-{}".format(self.spec_idx)
            else:
                cleaned_exp_tag = clean_tag(self.experiment_tag)
                exp_dir = format_unique_dir_name(cleaned_exp_tag, type='run')

            self.experiment_dir = exp_dir

        return self.experiment_dir

    def set_batch_run_spec(self, batch_tag: str, batch_dir: str) -> None:
        self.is_batch_spec = True
        self.batch_tag = batch_tag
        self.batch_dir = batch_dir
        return None

    def configure_batch_spec(self, batch_tag: str, batch_dir: str, spec_idx: int) -> str:
        self.set_batch_run_spec(batch_tag, batch_dir)
        self.spec_idx = spec_idx
        return self._reset_and_get_spec_run_path()

    def _reset_and_get_spec_run_path(self) -> str:
        self.experiment_path = None
        return self.get_spec_run_path()

    def get_spec_run_path(self) -> str:
        if self.experiment_path is None:
            root_path = os.path.relpath(self.root_experiment_dir)
            root_path = os.path.join(root_path, self.env_name)

            if self.batch_dir:
                root_path = os.path.join(root_path, self.batch_dir)

            spec_run_path = os.path.join(root_path, self.get_spec_run_dir())
            self.experiment_path = spec_run_path

        return self.experiment_path

    def setup_run_dir(self) -> None:
        exp_path = self.get_spec_run_path()

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        return None


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
class GenerateSaveLoadTrjExperimentSpec(PpoExperimentSpec):
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: float = field(default=0.5)


@dataclass
class PpoRudderExperimentSpec(ExperimentSpec):
    steps_by_epoch: Optional[int] = field(default=None)
    n_epoches: Optional[int] = field(default=None)
    hidden_dim: Optional[int] = field(default=None)
    n_hidden_layers: Optional[int] = field(default=None)
    n_trajectory_per_policy: Optional[int] = field(default=None)
    reward_delayed: bool = field(default=True)
    rew_factor: float = field(default=1.0)
    optimizer_weight_decay: float = field(default=0.0)
    optimizer_lr: float = field(default=1e-3)
    env_batch_size: Optional[int] = field(default=None)
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: float = field(default=0.5)
    selected_lstm_model_path: Optional[str] = field(default=None)
    lstm_model_name: Optional[str] = field(default=None)

    def __repr__(self) -> str:
        return super().__repr__()


@dataclass
class RudderLstmExperimentSpec(ExperimentSpec):
    env_batch_size: Optional[int] = field(default=None)
    model_hidden_size: Optional[int] = field(default=None)
    env_n_trajectories: Optional[int] = field(default=None)
    env_perct_optimal: float = field(default=0.5)
    n_epoches: Optional[int] = field(default=None)
    optimizer_weight_decay: float = field(default=0.0)
    optimizer_lr: float = field(default=1e-3)

    def __repr__(self) -> str:
        return super().__repr__()


# ::: Experiment Runner Organizer ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def format_unique_dir_name(cleaned_exp_tag: str, type: str = 'run') -> str:
    date_now = datetime.now()
    experiment_uuid = uuid.uuid1().int.__str__()
    unique_dir_name = "{}-{}-{}h{}--{}-{}-{}--{}".format(type, cleaned_exp_tag,
                                                         date_now.hour, date_now.minute,
                                                         date_now.day, date_now.month, date_now.year,
                                                         experiment_uuid)
    return unique_dir_name


def clean_tag(tag: str):
    cleaned_tag = ''
    if tag:
        if len(tag) != len(tag.replace(' ', '')):
            for x in tag.split(' '):
                cleaned_tag += x.capitalize()
        else:
            cleaned_tag = tag
    return cleaned_tag


def generate_batch_run_dir_name(batch_tag: str) -> str:
    batch_tag = clean_tag(batch_tag)

    batch_dir = format_unique_dir_name(batch_tag, type='batch')

    return batch_dir
