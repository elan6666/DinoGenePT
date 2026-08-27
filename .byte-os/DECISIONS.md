# Decision Log

## 2026-08-28

- Use official GenePT commit `3602699e7425a7be577771f8f07e218db6c79b9f`.
- Use Zenodo v2 record `10833191` and verify archive MD5
  `3f6ce4317e3a0091978ae5cb8fbf05a3`.
- Treat `text-embedding-3-large` over NCBI + UniProt as the latest official
  baseline and Doubao over the same text as GenePT-Seed.
- Select Table 1 gene-property tasks and Figure 2 gene-gene interaction as v0.
- Keep PPI and cell-level tasks outside v0 because their extra datasets do not
  improve the first embedding-backbone comparison enough to justify scope.
- Use local Git as the source of truth; server code is a checksum-verified
  mirror used only for data acquisition and execution.
- Never use the API key exposed in chat. A rotated key must be privately
  provisioned on the server before paid/live embedding calls.
- Implement a thin, tested compatibility layer because official GenePT is not
  an importable package and contains machine-specific notebook paths.
- Do not copy official notebook code wholesale despite reported permission;
  preserve provenance and implement only the small reusable evaluation kernel.

