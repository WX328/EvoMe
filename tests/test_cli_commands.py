from app.cli_commands import normalize_cli_input


def test_known_command_without_slash_is_normalized():
    assert normalize_cli_input("revision-signals") == "/revision-signals"
    assert normalize_cli_input("candidate-approve 3") == "/candidate-approve 3"
    assert normalize_cli_input("candidate-rebase 3") == "/candidate-rebase 3"
    assert normalize_cli_input("ancestor-revalidate") == "/ancestor-revalidate"
    assert normalize_cli_input("proposal-normalize 4") == "/proposal-normalize 4"
    assert normalize_cli_input("proposal-normalizations") == "/proposal-normalizations"
    assert normalize_cli_input("response-policy 5") == "/response-policy 5"


def test_natural_language_is_not_rewritten():
    assert normalize_cli_input("hypothesis 是什么意思？") == "hypothesis 是什么意思？"
    assert normalize_cli_input("帮我分析 revision signal") == "帮我分析 revision signal"


def test_existing_slash_command_is_unchanged():
    assert normalize_cli_input("/status") == "/status"


def test_learning_invalidate_command_normalizes_without_slash():
    assert normalize_cli_input("learning-invalidate 16 operational_input_misclassified") == "/learning-invalidate 16 operational_input_misclassified"
