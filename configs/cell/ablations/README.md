# Config-based ablations

The native registry is `dinogenept.cell.ablations.ablation_registry` (67 entries).
No separate model package or training/evaluation copy per ablation.

Inspect with `scripts/check_ablation_suite.py --list`. Pass each dataset's own
base recipe using `--recipe PATH --resolve ID` to print its variant. The resolver
preserves lineage and gives an `ablations/ID` output subdirectory when an output
path exists. Nothing launches implicitly.

Formal HVG runs additionally require the frozen training-only ID list and hash
receipt. Knowledge variants require real source-vector provenance. Tiny smoke
fixtures are not replacements for either input. See
`docs/ABLATION_IMPLEMENTATION_20260912.md` for commands and verification limits.
