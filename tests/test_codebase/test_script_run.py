# coding=utf-8
import pytest

pytestmark = pytest.mark.automated_test

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_script_run_ppo_PASS():
    from os import system

    # out = system("python -m Script_run_ppo --testSpec")
    out = system("ls .")
    out = system("python -m script.Script_run_ppo --testSpec")
    # out = system("cd script/ && ls . && python -m Script_run_ppo --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Module invocated from command line exited with error {}".format(out)

@pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_script_run_rudder_example_PASS():
    from os import system

    out = system("python -m Script_run_rudder_example --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Module invocated from command line exited with error {}".format(out)
