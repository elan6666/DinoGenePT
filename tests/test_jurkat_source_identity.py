import numpy as np
import pytest

ad = pytest.importorskip('anndata')
h5py = pytest.importorskip('h5py')
pd = pytest.importorskip('pandas')
from test_gene_identity import index, snapshot  # noqa: E402,F401

from dinogenept.jurkat_identity import load_jurkat_source_ids, read_annotation, source_id_record  # noqa: E402
from dinogenept.provenance import digest_file  # noqa: E402


def test_source_identity_preserves_conflict_without_relaxing_generic_resolver(snapshot):  # noqa: F811
    identity = index(snapshot)
    row = source_id_record(identity, 'A', 'ENSG00000000002.1', 'fixture.var.index')
    assert row['standard_ensembl_id'] == 'ENSG00000000002'
    assert row['annotation_conflict']
    assert row['original_label'] == 'A' and row['approved_symbol'] == 'B'
    assert identity.standardize('A', 'ENSG00000000002')['standard_ensembl_id'] is None
    row = source_id_record(identity, 'unknown', 'ENSG00000009999', 'fixture.obs.gene_id')
    assert row['standard_ensembl_id'] == 'ENSG00000009999'
    with pytest.raises(ValueError, match='human Ensembl'):
        source_id_record(identity, 'A', 'not-an-id', 'fixture')


def fixture_files(root, *, conflicting=False):
    (root / 'canonical').mkdir()
    (root / 'source').mkdir()
    cp, sp = root / 'canonical/adata.h5ad', root / 'source/raw.h5ad'
    obs = pd.DataFrame({'gene': ['A', 'B', 'B'],
                        'gene_id': ['ENSG00000000001', 'ENSG00000000002',
                                    'ENSG00000000003' if conflicting else 'ENSG00000000002']},
                       index=['c1', 'c2', 'c3'])
    original = pd.DataFrame({'gene_name': ['A']}, index=['ENSG00000000001'])
    canonical = pd.DataFrame({'gene_name': ['A', 'B']},
                             index=['ENSG00000000001', 'graph_only::B'])
    ad.AnnData(np.zeros((3, 1)), obs=obs, var=original).write_h5ad(sp)
    ad.AnnData(np.zeros((3, 2)), obs=obs, var=canonical).write_h5ad(cp)
    return {'canonical_adata_sha256': digest_file(cp)}, {
        'filename': 'raw.h5ad', 'source_sha256': digest_file(sp)}


def test_exact_axis_and_graph_only_guide_identity(tmp_path):
    canonical, source = fixture_files(tmp_path)
    axis, targets, proofs = load_jurkat_source_ids(tmp_path, canonical, source, ['A', 'B'])
    assert axis['A'][0] == targets['A'] == 'ENSG00000000001'
    assert axis['B'] == ('ENSG00000000002', 'source.obs.gene_id:graph_only_target')
    assert all(digest_file(path) == checksum for path, checksum in proofs.items())
    with pytest.raises(ValueError, match='exact graph axis'):
        load_jurkat_source_ids(tmp_path, canonical, source, ['B', 'A'])
    source['source_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='checksum'):
        load_jurkat_source_ids(tmp_path, canonical, source, ['A', 'B'])


def test_ambiguous_original_targets_fail_closed(tmp_path):
    canonical, source = fixture_files(tmp_path, conflicting=True)
    with pytest.raises(ValueError, match='Conflicting source perturbation IDs'):
        load_jurkat_source_ids(tmp_path, canonical, source, ['A', 'B'])


def test_legacy_h5ad_category_references(tmp_path):
    with h5py.File(tmp_path / 'legacy.h5ad', 'w') as f:
        group = f.create_group('obs')
        cats = group.create_dataset('categories', data=['A', 'B'], dtype=h5py.string_dtype())
        node = group.create_dataset('gene', data=[1, 0, -1])
        node.attrs['categories'] = cats.ref
        assert list(read_annotation(group, 'gene')) == ['B', 'A', None]
        node[0] = 3
        with pytest.raises(ValueError, match='category codes'):
            read_annotation(group, 'gene')
