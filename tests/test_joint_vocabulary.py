import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('torch')
from test_gene_identity import index, snapshot  # noqa: E402,F401

spec = importlib.util.spec_from_file_location(
    'joint_vocabulary', Path(__file__).resolve().parents[1] / 'scripts/build_50k_jurkat_vocabulary.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_alias_resolution_and_unresolved_retention(snapshot):  # noqa: F811
    identity = index(snapshot)
    assert module.token_identity(identity, 'OLD_A')[0] == module.token_identity(identity, 'ENSG00000000001')[0]
    key, report = module.token_identity(identity, 'SHARED')
    assert key == 'unresolved:Jurkat:SHARED' and report['status'] == 'ambiguous'
    assert module.token_identity(identity, 'ENSG00000009999')[0] == 'ENSG00000009999'
    assert module.token_identity(identity, 'unknown')[0] == 'unresolved:Jurkat:unknown'


def test_refuses_overwriting_old_bundle(tmp_path):
    with pytest.raises(FileExistsError):
        module.build(None, None, None, None, tmp_path)
