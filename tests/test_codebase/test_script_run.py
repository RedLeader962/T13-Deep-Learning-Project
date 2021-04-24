# coding=utf-8
import pytest

pytestmark = pytest.mark.automated_test

def command_line_test_error_msg(out):
    return "Module invocated from command line exited with error {}".format(out)


# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_script_run_ppo_PASS():
    from os import system

    out = system("python -m Script_run_ppo --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_script_run_rudder_example_PASS():
    from os import system

    out = system("python -m Script_run_rudder_example --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)


# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_Script_generate_save_load_trajectories_PASS():
    from os import system

    out = system("python -m Script_generate_save_load_trajectories --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_Script_run_LSTM_PASS():
    from os import system

    out = system("python -m Script_run_LSTM --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_Script_run_LSTM_to_LSTMCell_PASS():
    from os import system

    out = system("python -m Script_run_LSTM_to_LSTMCell --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_Script_run_LSTMCell_PASS():
    from os import system

    out = system("python -m Script_run_LSTMCell --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_Script_run_LSTMCell_PASS():
    from os import system

    out = system("python -m Script_run_LSTMCell --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

# @pytest.mark.skip(reason="Mute for now")
def test_command_line_invocation_Script_save_load_LSTM_PASS():
    from os import system

    out = system("python -m Script_save_load_LSTM --testSpec")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, command_line_test_error_msg(out)

