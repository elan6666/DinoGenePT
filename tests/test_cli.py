import json

import numpy as np

from dinogenept.cli import build_parser, main
from dinogenept.vectors import save_npz


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


def test_embedding_parser_uses_verified_agent_plan_batch_default():
    args = build_parser().parse_args(
        [
            "embed",
            "--texts",
            "texts.json",
            "--checkpoint",
            "checkpoint.sqlite3",
            "--output",
            "vectors.npz",
        ]
    )
    assert args.batch_size == 10


def test_vector_audit_can_require_exact_case_complete_universe(tmp_path, capsys):
    vectors = tmp_path / "vectors.npz"
    genes = tmp_path / "genes.txt"
    save_npz(
        vectors,
        {"ABC": np.ones(2), "Abc": np.ones(2)},
        "demo",
        uppercase_genes=False,
    )
    genes.write_text("ABC\nAbc\n", encoding="utf-8")
    assert (
        main(
            [
                "audit-vectors",
                "--vectors",
                str(vectors),
                "--genes",
                str(genes),
                "--preserve-gene-case",
                "--expected-dimension",
                "2",
                "--require-complete",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["coverage"]["requested"] == 2
    assert report["coverage"]["found"] == 2
