# Active Byte Auto state

- Goal: deliver the extensible DinoGenePT perturbation package, implement all
  planned config-only ablations, and prove every path with a one-epoch server
  smoke without using the occupied GPU.
- Local Git is the source of truth. Server execution root is
  `/data/yilangliu/GenePT-Seed`; no GraD-Pert source or run was modified.
- One model registry entry, `dinogenept`, now supports condition-bag DINO,
  delta-iBOT, sparse source-only locals, Base-only inference, attention
  residuals, Dense/MoE adapters, quantile routing, SiTU, KoLeo, and hybrid KDA.
- Dataset adapters/configs are separate for Adamson, Norman, Replogle K562, and
  Replogle RPE1. All models use the same condition-macro evaluator and output
  hierarchy.
- The 21-row Adamson-mini matrix completed on physical GPU 0 only; physical GPU
  1 remained occupied by the user's unrelated GraD-Pert process. Peak allocated
  memory was 66--79 MiB.
- Matrix identity is fixed by dataset/source/split/prior/checkpoint hashes,
  config SHA-256, executable source SHA-256, seed, and evaluator. All 21 model
  checkpoints and result artifacts passed hash verification; an idempotent
  recheck reused 21/21 rows.
- Full attention passed the permutation gate (`5.96e-7 <= 1e-5`). Hybrid KDA is
  explicitly order-sensitive (`0.20075`) and remains an experimental ablation.
- Local verification: 61 passed, 1 Torch-dependent skip; server verification:
  75 passed, Ruff passed, wheel/sdist built, both CLIs passed.
- Compact evidence is in `docs/results/dinogenept/SMOKE_VALIDATION.json`. Raw
  datasets, generated runs, checkpoints, and logs remain server-only.
- The smoke proves functionality only. No full GEARS/scGPT reproduction, real
  source-only knowledge embedding experiment, or biological model comparison
  has been run.
- User-owned untracked `uv.lock` remains excluded. The pre-existing untracked
  `.byte-os/LESSONS.md` is preserved and not staged.

## Next scientific step

Materialize the four real source-only priors, freeze an official Adamson/Norman
condition manifest, then run a preregistered multi-seed ladder from supervised
to Base DINO, delta-iBOT, dynamic locals, and semantic negative controls. Do not
select an architecture from the one-epoch smoke metrics.
