# coding=utf-8
import pytest

from script.general_utils import is_run_on_a_teamcity_continuous_integration_server

pytestmark = pytest.mark.automated_test

is_run_on_TeamCity_CI_server = is_run_on_a_teamcity_continuous_integration_server()

@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute for now")
def test_command_line_invocation_script_run_ppo_PASS():
    from os import system

    # out = system("ls .")
    # out = system("python -m Script_run_ppo --testSpec")
    # out = system("cd script/ && ls . && python -m Script_run_ppo --testSpec")
    out = system("python -m script.Script_run_ppo --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Module invocated from command line exited with error {}".format(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute for now")
def test_command_line_invocation_script_run_rudder_example_PASS():
    from os import system

    # out = system("python -m Script_run_rudder_example --testSpec")
    # out = system("python -m script.Script_run_rudder_example --testSpec")
    out = system("python -m script.Script_run_LSTM --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Module invocated from command line exited with error {}".format(out)
