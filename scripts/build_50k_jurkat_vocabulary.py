"""Build additive identity-audited vocabulary; never read Jurkat expression X.

Reads exact GraD-Pert canonical axes/split, not a reconstructed split. Unresolved
names stay in a separate audit with token -1, never a fabricated vocabulary ID.
Output is a separate vocabulary bundle, not an in-place dataset migration.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dinogenept.cell.genecompass_data import GeneCompassDataset
from dinogenept.cell.sampling import CropConfig
from dinogenept.gene_identity import GeneIdentityIndex
from dinogenept.provenance import atomic_write_json, digest_file


def token_identity(index, label, *, trusted_ensembl=False):
    row = index.standardize(label, trusted_ensembl=trusted_ensembl)
    return row['standard_ensembl_id'], row


def build(manifest, jurkat, hgnc, hgnc_sha256, output):
    if output.exists():
        raise FileExistsError(output)
    paths = [manifest, hgnc, jurkat / 'canonical/graph_gene_ids.txt',
             jurkat / 'canonical/expression_gene_ids.txt', jurkat / 'manifests/split.json',
             jurkat / 'manifests/canonical.json']
    hashes = {str(p.resolve()): digest_file(p) for p in paths}
    identity = GeneIdentityIndex(hgnc, expected_sha256=hgnc_sha256)
    data = GeneCompassDataset(manifest, CropConfig())
    data.verify()
    if len(data) != 50000:
        raise ValueError('Require the actual complete 50k dataset')
    vocab_path = data._path(data.manifest['vocabulary']['path'])
    old = json.loads(vocab_path.read_text())
    counts = np.zeros(len(old) + 1, dtype=np.int64)
    for i in range(len(data.shards)):
        ids, _ = data._read(i)
        counts += np.bincount(ids.reshape(-1), minlength=len(counts))
    observed = [old[i - 1] for i in range(1, len(counts)) if counts[i]]
    graph = paths[2].read_text().splitlines()
    expression = paths[3].read_text().splitlines()
    split = json.loads(paths[4].read_text())
    canonical = json.loads(paths[5].read_text())
    if split['dataset_id'] != 'nadig_jurkat' or canonical['state'] != 'canonical_ready':
        raise ValueError('Wrong or unready canonical Jurkat data')
    if digest_file(paths[4]) != canonical['split_manifest_sha256']:
        raise ValueError('Canonical split checksum differs')
    for labels, field in [(graph, 'n_graph_genes'), (expression, 'n_expression_genes')]:
        if len(labels) != canonical[field] or len(set(labels)) != len(labels) or '' in labels:
            raise ValueError('Invalid canonical gene axis')
    conditions = [split[k + '_conditions'] for k in ('train', 'val', 'test')]
    if len(set(sum(conditions, []))) != sum(map(len, conditions)):
        raise ValueError('Conditions overlap across splits')
    targets = sorted({g for c in sum(conditions, []) for g in c.split('+')
                      if g != split['control_condition_id']})
    sources = {'50k_observed': observed, 'jurkat_graph': graph,
               'jurkat_expression': expression, 'jurkat_targets': targets}
    entries, records = {}, {}
    for source, labels in sources.items():
        records[source] = []
        for label in labels:
            key, row = token_identity(identity, label, trusted_ensembl=source == '50k_observed')
            record = {**row, 'token_key': key}
            records[source].append(record)
            if key is None:
                continue
            item = entries.setdefault(key, {'sources': set(), 'original_labels': set()})
            item['sources'].add(source)
            item['original_labels'].add(label)
    union = sorted(entries)
    lookup = {key: i + 1 for i, key in enumerate(union)}
    mappings = {s: [lookup.get(r['token_key'], -1) for r in rows] for s, rows in records.items()}
    # A reduced union only maps observed old tokens. Unobserved rows map to -1,
    # even if their identity happens to be reintroduced by Jurkat.
    old_to_new = [-1] * (len(old) + 1)
    old_to_new[0] = 0
    old_lookup = {label: i + 1 for i, label in enumerate(old)}
    for label, token in zip(observed, mappings['50k_observed'], strict=True):
        old_to_new[old_lookup[label]] = token
    collision_cells = 0
    for i in range(len(data.shards)):
        ids, _ = data._read(i)
        mapped = np.asarray(old_to_new)[ids]
        if (mapped < 0).any():
            raise ValueError('Observed pretraining token missing from union')
        ordered = np.sort(mapped, axis=1)
        collision_cells += int(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] > 0)).any(1).sum())
    unresolved = {s: dict(Counter(r['status'] for r in rows)) for s, rows in records.items()}
    duplicates = {s: {str(k): n for k, n in Counter(ids).items() if n > 1 and k > 0}
                  for s, ids in mappings.items()}
    for p in paths:
        if digest_file(p) != hashes[str(p.resolve())]:
            raise ValueError('Source changed during build')
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        atomic_write_json(output / name, value)
    write('50k_observed_vocabulary.json', observed)
    write('vocabulary.json', union)
    write('old_to_new.json', old_to_new)
    write('source_token_ids.json', mappings)
    write('identity_audit.json', records)
    write('unresolved.json', {s: [r for r in rows if r['token_key'] is None] for s, rows in records.items()})
    write('token_sources.json', {key: {k: sorted(v) for k, v in item.items()} for key, item in entries.items()})
    pre_keys = {r['token_key'] for r in records['50k_observed']}
    receipt = dict(schema='dinogenept.independent-vocabulary.v2',
                   old_vocabulary_genes=len(old), observed_50k_genes=len(observed),
                   union_genes=len(union), embedding_rows=len(union) + 1,
                   jurkat_only_identities=len(set(union) - pre_keys),
                   source_counts={s: len(rows) for s, rows in records.items()},
                   split_counts=dict(zip(('train', 'val', 'test'), map(len, conditions), strict=True)),
                   identity_status=unresolved, duplicate_identity_tokens=duplicates,
                   unmapped_rows={s: ids.count(-1) for s, ids in mappings.items()},
                   pretraining_collision_cells=collision_cells, inputs_sha256=hashes,
                   vocabulary_source_sha256=digest_file(vocab_path),
                   token_policy='Ensembl primary; HGNC verification; unresolved names audited as -1',
                   downstream_values_read=False, active_dataset_rewritten=False,
                   readiness='identity_audit_required_before_dataset_migration',
                   source_sha256=digest_file(Path(__file__)))
    receipt['outputs_sha256'] = {p.name: digest_file(p) for p in output.glob('*.json')}
    write('receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'jurkat', 'hgnc', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hgnc-sha256', required=True)
    build(**vars(parser.parse_args()))
