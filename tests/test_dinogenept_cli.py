import json

from dinogenept.cli import main


def test_registry_cli_is_lazy_and_machine_readable(capsys):
    assert main(["registry"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == ["dinogenept"]
    assert payload["evaluators"] == ["perturbation"]
