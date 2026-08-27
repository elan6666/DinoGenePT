---
created_at: 2026-08-28T20:12:00+08:00
evidence_source: internal review and official model documentation
originating_review: review-1
---

# Evidence used

Review-1 manifest finding and official documentation that
`doubao-embedding-vision` returns 2048-dimensional dense vectors.

# Hypothesis and metric

A hard 2048 gate plus deterministic text fingerprint will fail closed on model
or input drift. Metric: wrong dimensions are rejected and valid mock runs write
a complete sidecar.

# Diagnosis

An NPZ alone cannot prove which text/model produced it.

# Changes planned

Add dimension validation, manifest fields, fingerprint, and regression tests.

# Changes made

Implemented the planned controls without storing request text or credentials in
the manifest.

# Verification

Server tests cover resume, valid sidecar, and wrong-dimension rejection.

# Remaining issues

No server key, so live response shape is not yet observed.

# Next decision

Proceed to result-lineage iteration.
