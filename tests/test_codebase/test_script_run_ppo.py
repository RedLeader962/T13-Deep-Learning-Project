# coding=utf-8
import pytest

pytestmark = pytest.mark.integration_ppo_logger

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_script_run_ppo_PASS():
    from os import system

    out = system("python -m Script_run_ppo")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Module invocated from command line exited with error {}".format(out)
