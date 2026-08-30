# Confirmed project lessons

## 2026-08-30: Keep architectural priors leakage-safe and missingness-native

- A teacher or adapter must not encode expression, perturbation outcomes, or
  control cells from an evaluation dataset unless that access is explicitly
  part of a separately named transductive protocol. For strict unseen-cell or
  unseen-perturbation claims, all evaluation cells and outcomes remain outside
  prior training.
- External text/ontology/graph knowledge is optional per gene. The complete
  NCBI+UniProt-compatible base expert is mandatory; a missing GO, Reactome,
  SIGNOR, HPA, or other expert is omitted rather than represented by invented
  text, neighbour propagation, a learned missing vector, or zero imputation.
- A fusion model must normalize only across experts actually present and return
  the base representation exactly when no optional expert exists. Report
  performance by source-availability strata and include a mask-only control to
  detect annotation-density or publication-bias shortcuts.
- Graph and relation sources require benchmark-pair overlap audits and a
  benchmark-sanitized variant before supporting inductive or leakage-robust
  claims.

## 2026-08-31: Keep DINOcell local knowledge views source-only

- In the proposed DINO-style multi-view perturbation training, each local view
  is a complete conditional Student input built from control-cell tokens plus
  one independently embedded knowledge source. For example,
  `GenePT-GO = DoubaoEmbed(GO-EXP only)`; it is not Base+GO text and not a Base
  residual view.
- Base remains its own mandatory Student view and the only perturbation prior
  used at primary inference. Optional GO, Protein, Pathway, and HPA views are
  train-time source-specific views and are omitted when that source is absent.
- Do not silently duplicate NCBI+UniProt content inside every local corpus.
  Repeating Base across views can make the self-distillation task trivial and
  obscure whether source-specific knowledge contributes.

## 2026-08-31: Separate cell pretraining, perturbation views, and DINOv2 losses

- An ordinary unperturbed atlas cell has no perturbation target. Do not attach
  a GenePT source token to such a cell and train it as if it represented a
  perturbation. Atlas pretraining uses expression-derived global views and
  masked gene-token prediction; source-specific perturbation views begin only
  on training examples with an observed perturbation condition.
- DINO/iBOT and KoLeo serve different roles. iBOT masks Student expression
  values while preserving gene identities and matches the unmasked Teacher at
  the same gene positions. KoLeo acts only on normalized Student global cell or
  perturbation representations across distinct conditions in a batch.
- Base, GO, Protein, and Pathway views of the same perturbation are positives.
  Never use KoLeo or another repulsion loss to separate those source views;
  source discrimination belongs in a negative-control probe, not the training
  objective.

## 2026-08-31: Distil GEARS perturbations at condition level

- GEARS control and post-perturbation cells are not paired observations of the
  same biological cell. Scouter-style random control sampling is valid as a
  Monte Carlo training strategy when the control is resampled for each
  perturbed cell and predictions are averaged across many controls at
  inference; never describe those samples as biological pairs. For DINO, pool
  teacher and student logits over a small same-condition bag before matching so
  random cell noise is not treated as view-invariant identity.
- A gene-token iBOT target can be solved from a visible static gene identity.
  Use a condition-level post-minus-control token target together with masked
  expression reconstruction, and include a shuffled-expression shortcut test.
- Knowledge availability is dynamic, but the amount of auxiliary gradient must
  not grow with annotation count. Always train the Base view and sample at most
  one optional source per condition step with source-balanced scheduling; never
  pad a missing source.
- KoLeo is not a default loss for perturbation conditions because functionally
  similar perturbations should be allowed to remain close. Treat it as a
  zero-included validation-only ablation and report representation-collapse
  diagnostics before claiming a benefit.

## 2026-08-31: Name the Scouter prior swap honestly and keep it Base-only

- The user-selected comparison is `Scouter + GenePT-Seed Base`, where Base is
  exactly NCBI+UniProt text embedded by Doubao. It is not an exact reproduction
  of the official Scouter paper baseline and must never be labelled as one.
- Call the pinned `scouter-learn` package for the Scouter predictive network and
  training method; do not copy or reimplement the model. Freeze the common
  GraD-Pert split, controls, seeds, gene order, and metrics around that adapter.
- For this prior-swap comparison, reject GO, Protein, Pathway, HPA, enriched
  progressive vectors, and silent missing-target deletion. Base must cover
  every perturbation target or the run stops before model construction.
