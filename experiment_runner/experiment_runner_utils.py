# coding=utf-8
import random
from collections import namedtuple
from copy import deepcopy
from typing import Callable, Dict, List, Union

from experiment_runner.experiment_spec import ExperimentSpec
from experiment_runner.parameter_search_map import RudderLstmParameterSearchMap, PpoRudderParameterSearchMap

CONSOL_WIDTH = 85

ExperimentResults = namedtuple('ExperimentResults', ['loss_train', 'loss_test'])


def check_repository_pulled_to_local_drive_ok() -> None:
    print("hellooooowwwwwww RedLeader962!")
    return None


def print_experiment_header(name: str, length) -> None:
    lll = length - len(name)
    print(f"\n\n:::: {name}", ":"*lll, '\n')
    return None


def print_end_experiment_header(name: str, length) -> None:
    lll = length - len(name)
    print(
        '\n',
        ":"*lll,
        f"{name} ::::\n",
        ":"*(length + 6),
        f"\n\n",
        )
    return None


def execute_experiment_plan(exp_specs: List[ExperimentSpec], script_fct: Callable) -> Dict[str, ExperimentSpec]:
    """
    Execute a planned experimentation over a function `script_fct` taking a `ExperimentSpec` dataclass as argument.

    Pre condition:
     1. Every `ExperimentSpec` field must be instanciated
     2. `script_fct` must be callable and take an `ExperimentSpec` instance as first argument

    :return: a dictionary of `key=spec_idx`  `value=ExperimentSpec` with appended results
    """
    specs_w_result = dict()
    exp_len = len(exp_specs)

    # ... Check pre-condition ..........................................................................................
    if not isinstance(exp_specs, list):
        exp_specs = [exp_specs]
    for each_spec in exp_specs:
        assert isinstance(each_spec, ExperimentSpec)
    assert callable(script_fct)

    # ... Start experiment .............................................................................................
    each_spec: ExperimentSpec
    for spec_id, each_spec in enumerate(exp_specs, start=1):
        print_experiment_header(name=f'Start experiment {spec_id}/{exp_len}', length=CONSOL_WIDTH)

        try:
            each_spec.is_batch_spec = True
            each_spec.spec_idx = spec_id
            print(each_spec)

            exp_result: ExperimentResults = script_fct(each_spec)
            each_spec.results = exp_result
            specs_w_result[f'{spec_id}'] = each_spec

        except AssertionError as e:
            print(e)

    print_end_experiment_header(f'{exp_len} experimentation DONE', CONSOL_WIDTH)
    return specs_w_result


def execute_parameter_search(exp_spec: Union[RudderLstmParameterSearchMap, PpoRudderParameterSearchMap],
                             script_fct: Callable,
                             exp_size: int,
                             start_count_at: int = 1,
                             consol_print: bool = True) -> Dict[str, ExperimentSpec]:
    """
    Execute a randomnized parameter search experimentation over a function `script_fct` taking a
    single `RudderLstmParameterSearchMap` or `PpoRudderParameterSearchMap` dataclass as argument.

    Randomnized specification will be produce for every `RudderLstmParameterSearchMap` or `PpoRudderParameterSearchMap`
    field taking callable argument.

    ex:
        >>> test_spec = RudderLstmParameterSearchMap(
        >>>    env_batch_size=4,
        >>>    model_hidden_size=lambda: random.choice([11, 22]),
        >>>    )

    Pre condition:
     1. Require that `RudderLstmParameterSearchMap` or `PpoRudderParameterSearchMap` has at least one field
        with a callable argument
     2. `script_fct` must be callable and take an `ExperimentSpec` instance as first argument

    :return: a dictionary of `key=spec_idx`  `value=ExperimentSpec` with appended results
    """

    specs_w_result = dict()
    stop_at = start_count_at + exp_size

    # ... Check pre-condition ..........................................................................................
    assert isinstance(exp_spec, ExperimentSpec)
    assert callable(script_fct)

    # ... Start experiment .............................................................................................
    for idx in range(start_count_at, stop_at):
        if consol_print:
            print_experiment_header(name=f'({idx}/{exp_size}) Start experiment {idx + start_count_at - 1}',
                                    length=CONSOL_WIDTH)

        try:
            exp_spec.is_batch_spec = True
            exp_spec.randomnized_spec()
            this_spec = deepcopy(exp_spec)
            this_spec.spec_idx = idx

            if consol_print:
                print(this_spec)

            exp_result: ExperimentResults = script_fct(this_spec)
            this_spec.results = exp_result

            specs_w_result[f'{idx}'] = this_spec

        except AssertionError as e:
            print(e)

    if consol_print:
        print_end_experiment_header(f'{exp_size} experimentation DONE', CONSOL_WIDTH)

    # if consol_print:
    #     print(specs_w_result)
    return specs_w_result
