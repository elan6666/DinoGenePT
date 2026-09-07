# GeneCompass source audit — 2026-09-07

The user selected the human500k published corpus and requested50k/500k/5M
downloads. This selection is retained. Archive contents are NOT yet audited;
the following evidence describes upstream example code, not verified contents
of the downloaded release. Do not silently switch datasets or launch training.

## Pinned source

- Repository: https://github.com/xCompass-AI/GeneCompass
- Revision: `59e5e485cb940038a7db040c2f31c8a2ce76ba53`
- File: `preprocess/preprocess.py`
- SHA256: `1c26f715b377f5a71a91ab515b63df50dfaca17f26a4eb72290bb450306d5057`
- Read as source only, never executed or imported.

## Observed behavior

1. `Normalized` divides each gene by a supplied gene-specific median. The
   library-total normalization expression is commented out; the active path
   does NOT perform the advertised parameter's total-count normalization.
   Subsequent `log1p` uses base2. These are continuous transformed values,
   not recoverable raw counts without the necessary upstream information.
2. `tokenize_cell` selects nonzero values and sorts descending.
   `rank_value` truncates to2048 and pads shorter sequences with zeros.
   Omitted genes cannot be treated as biologically zero or reconstructed.
3. `transfor_out` emits only `input_ids` int32 sequences, `values` float32
   sequences, `length` int16 sequences, and `species` int16 sequences.
   It does not export study, donor, or original cell identifiers.
4. Token IDs require the matching upstream gene-token dictionary. They must
   not be assumed identical to our Census Ensembl vocabulary indices.

## Archive audit consequences

- Inspect actual Arrow/schema/sidecars after archive integrity verification.
  The release may contain metadata beyond what this example exports.
- If metadata is absent, do not claim donor/study-disjoint validation or
  absence of downstream contamination. Seek a source mapping or ask the user
  about a clearly disclosed protocol change; do not manufacture identifiers.
- If top2048 is confirmed, crops operate only on the available observed gene
  subset. Do not re-normalize using its truncated sum or fill excluded genes
  as real zero-expression observations.
- A continuous-value loader must have an explicit scale contract, not bypass
  the existing raw-count validator. New dataset-specific tests and a frozen
  split/config are required before formal pretraining.
