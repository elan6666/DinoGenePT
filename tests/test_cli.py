import json

from genept_seed.cli import build_parser, main


def test_doctor_never_prints_key(monkeypatch, capsys):
    secret = "do-not-print-this"
    monkeypatch.setenv("ARK_API_KEY", secret)
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert secret not in output
    assert json.loads(output)["ark_api_key_present"] is True


def test_benchmark_parser_requires_explicit_normalization():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "ggi",
            "--name",
            "demo",
            "--vectors",
            "vectors.npz",
            "--data",
            "ggi",
            "--output",
            "result.json",
            "--genes",
            "common.txt",
        ]
    )
    assert args.normalize is False
    assert args.genes.name == "common.txt"
