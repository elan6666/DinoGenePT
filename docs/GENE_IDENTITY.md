# Unified human gene identity

## Primary-key standard (2026-09-09)

New vocabularies use versionless Ensembl Gene IDs, not HGNC IDs or gene names,
as primary keys. `GeneIdentityIndex.standardize` shares the existing HGNC
approved-name / previous-name / alias priority and conflict checks. A name-only
match needs exactly one valid associated Ensembl ID. Invalid, ambiguous, missing
and conflicting mappings produce no token key, not a guessed replacement.

An independently hash-verified source Ensembl ID may be retained even if absent
or ambiguous in this HGNC snapshot, with status `source_verified_hgnc_unresolved`.
The flag is explicit and applies only to ID-labelled records; it cannot bypass
a contradictory name/ID pair. This is source evidence, not proof of current
Ensembl annotation validity. Original labels,HGNC identity/symbol and sources
remain in the audit. The legacy `resolve`/knowledge opt-in behavior is preserved;
historical datasets and vector files are not automatically migrated.

The new bundle is `data/vocabulary/genecompass50k-jurkat-ensembl-v2` on the server.
It contains20685 Ensembl keys (+PAD0),19827 observed50k genes and858 added keys.
Unmapped Jurkat rows:graph98,expression95,targets1 (QARS). Lists overlap; do not
sum them as distinct genes. All unmapped rows retain source order and sentinel-1
in `source_token_ids.json` plus full `unresolved.json`. A duplicate axis identity
also remains (C20orf197/MIR646HG). Therefore the bundle is not training-ready.
No rows/columns/targets were deleted from source datasets; no expression values
were merged. The v1 mixed-key bundle remains historical and is superseded for
new vocabulary work, not overwritten. See [v2 receipt](results/JOINT_VOCABULARY_ENSEMBL_20260909.json).

Any future activation requires a complete mapping and duplicate-column policy,
then explicit dataset-ID remapping and vocabulary-bound configuration validation.
Neither-1 nor unresolved names may reach an embedding lookup.23 focused identity
and vocabulary tests passed for this revision; no training was launched.

Implemented as an **opt-in, non-destructive mapping layer**, not a rewrite of
training shards, vocabulary positions, expression values or old vector files.
All sources share one `GeneIdentityIndex`; there are no online lookups at load
time. The current snapshot is `data/hgnc/2026-08-28/hgnc_complete_set.txt`, SHA256
`0615a070f1628e6727953f67ad9248dd0f0ddbb16d41a7b40e06aa852fc3f448`.

HGNC's documented `hgnc_id`, approved `symbol`, `prev_symbol`, `alias_symbol`,
`status` and `ensembl_gene_id` fields supply identity evidence, not learned
features. See [official field definitions](https://www.genenames.org/help/statistics-and-downloads/).
Our conflict rules below are deliberately conservative engineering policy,
not a claim that HGNC specifies this exact algorithm.

## Rules

1. Keep original labels, original Ensembl IDs (including version), and column
   positions. Strip only a valid human ENSG numeric version for lookup; do not
   strip periods from symbols or mix human/mouse IDs.
2. Approved-name match precedes previous-name then alias match. Case folding is
   only an explicit human-name lookup aid; original labels are never overwritten.
3. If an ID is supplied, compare it with name evidence. A contradictory match,
   unknown supplied ID or ambiguous candidate set is unresolved, never guessed.
   A known unique ID may resolve an unrecognized name. Only Approved HGNC rows
   participate. Source snapshot hash is mandatory.
4. Different axis columns reaching the same HGNC ID remain separate in the
   report and fail the strict loader. No sum/average/drop of processed values.
   A dictionary containing two identical JSON keys also fails instead of losing
   the first entry during parsing.
5. Source-only GO/Protein/Pathway/HPA and TextBase use the same identity table.
   Same identity plus byte-identical text may reuse a deterministic existing
   key. Different texts retain every variant and require an explicit decision.
   Ambiguous source keys that could refer to a requested gene are errors, not
   permission to omit that local.
6. Distinguish unresolved identity, no resolved record in this source snapshot,
   conflicting text, and verified vector availability. A source record is not
   necessarily functional knowledge: old master identity-only text remains so.
7. Reuse vectors ONLY after existing corpus/model/dimension/text-fingerprint/
   vector-checksum checks pass. Identical-text alias keys with differing vectors
   fail. No implicit averaging, embedding API calls, zero fill or regeneration.
8. Missing optional sources yield no local; a combination needs every target's
   source. Missing TextBase fails. Output arrays retain the requested axis order.

## Entry points

`dinogenept data audit-gene-identities --hgnc SNAPSHOT --hgnc-sha256 SHA
--axis AXIS.json --sources SOURCES.json --output NEW_REPORT.json`

- `AXIS.json`: ordered string labels, or objects such as
  `{"label":"OLD_NAME","ensembl_id":"ENSG00000000001"}`.
- `SOURCES.json`: source-name to corpus-path JSON object. An empty object audits
  identity alone. Reports preserve every row; exit0 means audit completed,
  NOT that all inputs are safe. Inspect `safe_unique_axis`, counts and joins.
- `GeneIdentityIndex`, `SourceIdentityIndex` and `build_identity_report` are
  native package APIs. `IdentityKnowledgeBank.for_axis` returns row positions
  plus actual vectors (including an empty array for an absent optional source).

The existing `KnowledgeBank`/fine-tuning knowledge configuration can opt in:

```json
"identity": {
  "hgnc": "data/hgnc/2026-08-28/hgnc_complete_set.txt",
  "hgnc_sha256": "0615a070f1628e6727953f67ad9248dd0f0ddbb16d41a7b40e06aa852fc3f448",
  "axis": {"path": "PATH_TO_ORDERED_AXIS_JSON", "sha256": "ITS_SHA256"}
}
```

This is a schema example, not a runnable configuration. Other knowledge entries
(five source proofs, vectors, embedding manifests, model and width) remain
required. The axis hash and its exact label order must match the expression
axis; training callers cannot silently reorder it. Old configs retain their
exact-match behavior. **No running or pending experiment config was switched**:
the real audit below identifies issues to resolve first.

## Server audit, September 8

The additive campaign script `scripts/audit_current_gene_identities.py` audited
all current23113 vocabulary IDs, all17730 master labels, and the released
CellFM Adamson/Norman axes and targets. It verified input hashes before and
after. Full per-row reports stay server-side under `data/identity/2026-09-08-v3`;
[compact receipt](results/GENE_IDENTITY_AUDIT_20260908.json) is versioned.
v1 tested master-only corpus joins; v2 added the actual extended downstream
bundle; v3 also blocks ambiguous source identities. Earlier outputs are retained
as audit lineage, not production mappings or altered datasets.

| Scope | Rows | HGNC-resolved | Unresolved/ambiguous |
|---|---:|---:|---:|
| GeneCompass vocabulary | 23113 | 20698 | 2415 |
| Master labels | 17730 | 16157 | 1573 |
| Adamson axis | 1069 | 1068 | 1 |
| Norman axis | 1049 | 1049 | 0 |
| Adamson targets | 78 | 78 | 0 |
| Norman targets | 100 | 100 | 0 |

These are snapshot-specific matches, not evidence that unresolved genes are
invalid. Master labels have63 groups collapsing to the same identity, including
historical aliases/case pairs; nothing was merged. Its Base joins flag111 rows
with differing text variants and one row affected by an ambiguous source key.

Downstream joins use **their already extended1777-gene source-only bundle**,
not master-only text. Target TextBase coverage is78/78 and100/100. Optional
GO coverage is76/78 and95/100; omitted records remain omitted. Full Adamson
axis has one unresolved supplied ID for SOD2 (`ENSG00000112096`) in this HGNC
snapshot; do not override it using name alone. Norman Base covers1049/1049.

Identity matching is not pretraining-token coverage. Adamson has76/78 target
identities in the frozen pretraining vocabulary; Norman94/100. The missing
links are HYOU1, NEDD8; CDKN1C, CKS1B, DUSP9, KMT2A, RHOXF2 and SAMD1. An
extension/initialization policy is a separate model decision before fine-tuning,
not authorization to change the active500k pretraining vocabulary. This task
does not claim fine-tuning readiness or coverage of future datasets.

Verification:31 focused local tests and208 full server tests (CUDA hidden for
CPU tests), plus Ruff, native CLI help, wheel/sdist builds and input hash checks.
These include ambiguity, ID/name disagreement, versioned IDs, duplicate axes,
duplicate JSON keys, conflicting source texts, unchanged-vector row order,
optional source omission, frozen-axis mismatch and the existing loader opt-in.
