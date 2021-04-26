# coding=utf-8
from os import getenv

import argparse
from collections import namedtuple
from typing import Tuple

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    show_plot: bool


def is_automated_test() -> bool:
    """ Check if code is executed from a test suite """
    automated_test = False
    try:
        globals_pytestmark_ = globals()['pytestmark']
        if globals_pytestmark_.markname == 'automated_test':
            automated_test = True
    except KeyError:
        pass
    return automated_test


def check_testspec_flag_and_setup_spec(user_spec: ExperimentSpec,
                                       test_spec: ExperimentSpec) -> Tuple[ExperimentSpec, bool]:
    """
    Parse command line arg and check for unittest specific. Return test specification if --testSpec is raise
    :return: user_spec or test_spec
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testSpec', action='store_true', help='Use test spec')

    args = parser.parse_args()

    is_test_run = False
    spec = user_spec

    if args.testSpec:
        is_test_run = True
        spec = test_spec

    return spec, is_test_run


def is_test_run_under_teamcity_CI_check(set: bool, msg: str = 'Matplotlib') -> None:
    """
    Note: Required to switch off matplotlib plt.show() on TeamCity continuous intergation server

    :param set: True or not during pytest run on local machine
    :return: 'set' if it run locally, True otherwise
    """

    try:
        import teamcity as tc

        # if it return 'LOCAL' then it is not running on a TeamCity server
        tc_version = getenv('TEAMCITY_VERSION')

        if tc_version != 'LOCAL':
            print(f'>>> is running under teamcity TEAMCITY_VERSION={tc_version}   ==>   switching off {msg}')
            return True
        else:
            print(f'>>> TEAMCITY_VERSION={tc_version}')
            return set
    except ImportError:
        return set
