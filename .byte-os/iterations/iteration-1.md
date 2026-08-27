---
created_at: 2026-08-28T20:05:00+08:00
evidence_source: internal review and server download failures
originating_review: review-1
---

# Evidence used

GGI CDN stalls, a zero-byte partial file, and review-1 checksum finding.

# Hypothesis and metric

Pinning the Gene2vec commit and four SHA-256 values will make every accepted GGI
file byte-identical. Metric: four of four files validate on a repeated prepare.

# Diagnosis

Non-empty files alone are not proof of completeness.

# Changes planned

Use GitHub Contents API raw media, a fixed commit, atomic download, and hashes.

# Changes made

Implemented all planned controls and downloader fallback/retry behavior.

# Verification

Repeated server preparation validated all four expected SHA-256 values.

# Remaining issues

Live embedding provenance still needed.

# Next decision

Proceed to embedding safety and manifest iteration.
