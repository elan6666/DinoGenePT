# GraD-Pert Jurkat full-axis priors

This experiment uses the frozen Nadig Jurkat runtime axis, not the Gene2vec GGI
subset. The graph axis contains 2,809 exact, case-sensitive symbols and has
SHA-256 `e14dc759fa3744c552ecc7be82ce84ea50bff773c6cfdc89b53418e5350ff389`.

## Full-axis text corpus

The server artifact
`data/gradpert-jurkat/e14dc759fa37/NCBI_UniProt_HGNC_Jurkat_axis.json`
contains one non-empty text for every graph node. Its SHA-256 is
`3d15a4c7f36e5d93311564114f0b49167dc56986a7d0f16cdd90f391f3c58962`.

Source accounting is:

- 2,702 exact GenePT NCBI + UniProt texts;
- 61 old symbols mapped by HGNC to an approved symbol with an existing GenePT
  text;
- 32 HGNC identity descriptions where GenePT has no text;
- 14 minimal Ensembl archive identity descriptions where no current HGNC record
  exists.

The exact-axis mapping, HGNC complete set, and Ensembl archive bundle are pinned
in the manifest by SHA-256. No graph node is dropped, uppercased, or filled with
invented biological function.

## GO-EXP condition

`NCBI_UniProt_HGNC_GOEXP_Jurkat_axis.json` has SHA-256
`3e835f5e8e2f555c493387d608263734ce6f9609c11c0eae551546c23c87cfee`.
It uses the 2026-08-05 GO release, keeps EXP, IDA, IMP, IEP, HTP, HDA, HMP, and
HEP evidence, excludes `NOT`, and excludes interaction-derived IPI, IGI, and
HGI evidence from this GGI-adjacent text prior. At most eight non-redundant
terms per aspect are retained. GO text was added to 2,549 of 2,809 genes.

## Official GenePT comparator

The latest verified protein-aware official GenePT artifact is aligned to the
same axis in `genept_latest_model3_Jurkat_axis.npz`, SHA-256
`79a7033a3d5954a2450935b84e9661500e24fe2ab7e1ec5c4ce6c463b80d02e7`.
It provides 2,743 exact vectors and six HGNC-safe alias vectors. The remaining
60 rows contain zero prior and must be paired with a downstream learned-ID
residual. They are not claimed as official GenePT embeddings.

## Pending execution

GenePT-Seed and GenePT-Seed+GO vectors require the server-side Ark credential.
The server currently has no `ARK_API_KEY`; no vector API request or GraD-Pert
training run is represented as complete until those artifacts and their hashes
exist.
