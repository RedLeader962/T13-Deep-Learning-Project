# coding=utf-8

import pytest

pytestmark = pytest.mark.automated_test

def test():
    assert type() is type(pytest)

# def test_fail():
#    raise AssertionError


# @pytest.mark.skip(reason="Mute for now")
def test_server_setup_ci_deployement_master_branch_PASS():
    assert True
