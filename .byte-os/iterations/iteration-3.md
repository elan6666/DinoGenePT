---
created_at: 2026-08-28T20:22:00+08:00
evidence_source: internal review and final server verification
originating_review: review-1
---

# Evidence used

Review-1 lineage finding, server package-build stalls, and matched baseline runs.

# Hypothesis and metric

Explicit result receipts and build exclusions will make the repository auditable
without scanning or packaging generated data. Metrics: source package under
100 KB, all tests/lint pass, and result row has hashes and dependency versions.

# Diagnosis

Server mirrors lack `.git`, so default sdist discovery can include large runtime
artifacts. Earlier result rows also could not detect input drift.

# Changes planned

Record hashes/versions/seeds, exclude runtime paths, and add a dry-run sync tool.

# Changes made

Implemented all controls, plus common-gene evaluation and documentation.

# Verification

Server: 15 tests passed, Ruff passed, wheel 18,067 bytes, sdist 28,326 bytes,
GGI hashes passed, and final baseline lineage row was produced.

# Remaining issues

Only the credential-gated Doubao run and matched comparison remain.

# Next decision

Run final review; block only on private credential provisioning.
