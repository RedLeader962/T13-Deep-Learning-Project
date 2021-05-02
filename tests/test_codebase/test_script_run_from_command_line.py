# coding=utf-8
import pytest

from experiment_runner.test_related_utils import is_run_on_a_teamcity_continuous_integration_server

pytestmark = pytest.mark.automated_test

is_run_on_TeamCity_CI_server = is_run_on_a_teamcity_continuous_integration_server()


def command_line_test_error_msg(out):
    return "Module invocated from command line exited with error {}".format(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_script_run_ppo_PASS():
    from os import system

    # out = system("ls .")
    # out = system("python -m Script_run_ppo --testSpec")
    # out = system("cd script/ && ls . && python -m Script_run_ppo --testSpec")
    out = system("python -m script.Script_run_ppo --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_script_run_rudder_example_PASS():
    from os import system

    # out = system("python -m Script_run_rudder_example --testSpec")
    # out = system("python -m script.Script_run_rudder_example --testSpec")
    out = system("python -m script.Script_run_LSTM --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)



@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
# @pytest.mark.skip(reason="Just mute")
def test_command_line_invocation_Script_run_LSTM_PASS():
    from os import system

    out = system("python -m script.Script_run_LSTM --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_Script_run_LSTM_to_LSTMCell_PASS():
    from os import system

    out = system("python -m script.Script_run_LSTM_to_LSTMCell --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_Script_run_LSTMCell_PASS():
    from os import system

    out = system("python -m script.Script_run_LSTMCell --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_Script_run_LSTMCell_PASS():
    from os import system

    out = system("python -m script.Script_run_LSTMCell --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


@pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
def test_command_line_invocation_Script_save_load_LSTM_PASS():
    from os import system

    out = system("python -m script.Script_save_load_LSTM --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# @pytest.mark.skipif(is_run_on_TeamCity_CI_server, reason="Mute on CI server")
@pytest.mark.skip(reason="Won't fix")
def test_command_line_invocation_Script_run_ppo_with_rudder_PASS():
    from os import system

    out = system("python -m script.Script_run_ppo_with_rudder --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


# @pytest.mark.skipif(not is_run_on_TeamCity_CI_server, reason="Pour ne pas écraser les données dans le experiment/ dir")
@pytest.mark.skip(reason="Just mute")
def test_command_line_invocation_Script_generate_save_load_trajectories_PASS():
    from os import system

    out = system("python -m script.Script_generate_save_load_trajectories --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)