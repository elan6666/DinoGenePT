"""Read-only, hash-bound source identities; never load expression X.

Source feature/guide IDs define dataset identity, while HGNC supplies current
annotation. Historical label conflicts remain explicit, not silently renamed.
"""

from collections import defaultdict

import h5py
import numpy as np
from anndata.io import read_elem

from .gene_identity import stable_ensembl
from .provenance import digest_file


def read_annotation(group, name):
    """Decode legacy H5AD category references as well as modern categoricals."""
    node = group[name]
    values = read_elem(node)
    if 'categories' in node.attrs:
        categories = read_elem(node.file[node.attrs['categories']])
        if not np.issubdtype(values.dtype, np.integer) or (values < -1).any() or (values >= len(categories)).any():
            raise ValueError('Invalid legacy annotation category codes')
        return np.asarray([None if i == -1 else categories[i] for i in values], dtype=object)
    return values


def source_id_record(index, label, source_id, evidence):
    stable = stable_ensembl(source_id)
    if stable is None:
        raise ValueError('Source gene ID is not a human Ensembl gene ID')
    annotation = index.standardize(stable, trusted_ensembl=True)
    comparison = index.resolve(label, stable)
    return {**annotation, 'original_label': label, 'original_ensembl_id': source_id,
            'source_evidence': evidence, 'name_id_check': comparison,
            'annotation_conflict': comparison['status'] == 'conflict',
            'standardization_status': 'source_id_with_annotation_conflict'
            if comparison['status'] == 'conflict' else annotation['standardization_status']}


def load_jurkat_source_ids(root, canonical, source_manifest, graph):
    cp = root / 'canonical/adata.h5ad'
    sp = root / 'source' / source_manifest['filename']
    for path, expected in [(cp, canonical['canonical_adata_sha256']),
                           (sp, source_manifest['source_sha256'])]:
        if digest_file(path) != expected:
            raise ValueError(f'Source H5AD checksum differs: {path}')
    with h5py.File(cp, 'r') as f:
        var = read_elem(f['var'])
    if var['gene_name'].astype(str).tolist() != graph:
        raise ValueError('Canonical var does not match the exact graph axis')
    with h5py.File(sp, 'r') as f:
        original = read_elem(f['var'])
        # Explicitly only target annotation fields, no expression or outcomes.
        genes = read_annotation(f['obs'], 'gene')
        ids = read_annotation(f['obs'], 'gene_id')
    if not original.index.is_unique:
        raise ValueError('Duplicate original feature IDs')
    raw_features = {str(i): str(n) for i, n in zip(original.index, original.gene_name, strict=True)}
    targets = defaultdict(set)
    for label, gene in zip(genes, ids, strict=True):
        if stable_ensembl(str(gene)):
            targets[str(label)].add(stable_ensembl(str(gene)))
    ambiguous = {g: sorted(v) for g, v in targets.items() if len(v) != 1}
    if ambiguous:
        raise ValueError(f'Conflicting source perturbation IDs: {ambiguous}')
    target_map = {g: next(iter(v)) for g, v in targets.items()}
    axis = {}
    for raw, name in zip(var.index.astype(str), graph, strict=True):
        stable = stable_ensembl(raw)
        if stable:
            if raw not in raw_features:
                raise ValueError('Canonical feature missing from original source')
            original_name = raw_features[raw]
            if name not in {original_name, original_name + '__' + raw}:
                raise ValueError('Canonical feature label not traceable to original label')
            axis[name] = (stable, 'source.var.index+canonical.var.index')
        elif raw == 'graph_only::' + name and name in target_map:
            axis[name] = (target_map[name], 'source.obs.gene_id:graph_only_target')
        elif not raw.startswith('graph_only::'):
            raise ValueError('Unrecognized canonical feature identity')
    proofs = {str(cp.resolve()): canonical['canonical_adata_sha256'],
              str(sp.resolve()): source_manifest['source_sha256']}
    return axis, target_map, proofs
