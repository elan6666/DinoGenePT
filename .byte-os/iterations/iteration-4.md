---
created_at: 2026-08-28T19:40:00+08:00
evidence_source: server corpus manifests and SHA-256 audit
originating_review: scope correction
---

# Evidence used

Completed NCBI+UniProt corpus, pinned GO release, 10,870-gene GGI allowlist,
and both GO corpus manifests.

# Hypothesis and metric

Rebuilding GO from the completed corpus should preserve every selected GGI
base text while recording the correct extension lineage. Metrics: exact output
hash equality and complete selected-gene coverage.

# Diagnosis

The first GO manifest referenced the immutable official base rather than the
completed extension, although all added missing symbols lay outside GGI.

# Changes made

Rebuilt the bounded GO-EXP corpus with the completed extension as its explicit
base. The selected output remained byte-identical.

# Verification

Both selected GO corpora have SHA-256 `c66efcad...06f93`; the new manifest pins
base SHA-256 `e7c76d01...1d266`, 10,870 genes, and GO text for 9,597 genes.

# Remaining issues

None for corpus identity.

# Next decision

Verify API cache rows and materialize both complete embeddings.
