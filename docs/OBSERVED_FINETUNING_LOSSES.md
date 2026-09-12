# Observed fine-tuning: native adaptation, not upstream DINOcell parity

Pretraining losses, temperatures, EMA and loss ramp are unchanged.

Fine-tuning defaults now use
`0.5 * reconstruction + primary_dino + knowledge_dino + observed_dino`.
Knowledge is the mean over available non-primary sources, not a sum.
No fine-tuning loss ramp is introduced.

Optional observed iBOT and condition-filtered KoLeo are OFF by default. When
enabled their weights are 0.5 and 0.1, respectively. Registry overlays
`F0-OBSERVED`, `F1-OBSERVED`, `F2-OBSERVED`, `F3-OBSERVED` select neither,
iBOT only, KoLeo only, and both. These overlays do not authorize formal runs.

## Training-only sampling

Fixed train/validation/test membership precedes pairing. Each primary training
post cell remains visited exactly once per epoch. Teacher A and observed B are
sampled without replacement from disjoint subsets of the same condition and
cell type, further stratified by available donor, dose and time fields. The
prediction/control population and held-out evaluation protocol are unchanged.
Missing donor metadata is not inferred: input_identity reports the selected
fields and whether donor matching is guaranteed. Unrecognized donor field names
need explicit mapping; they must not be claimed as donor-matched.

Singleton groups have no observed branch; primary/control/knowledge training
continues. Model entry validates unique, disjoint observed/teacher IDs. Dataset
entry validates membership and matching metadata. Teacher cell-level targets
remain the mean of per-cell probabilities within the matched group, not a
random A-to-B one-to-one correspondence. One forward contains one such group.

## Optional iBOT

Observed B uses capped uniform gene sampling; zeros remain eligible. When iBOT
is enabled, 50% of B cells are selected (stochastic rounding for odd bag sizes);
20% of effective gene positions are hidden (floor, minimum one). Expression is
not binned. Teacher receives exactly the same B cells, gene order and expression,
with only the hidden mask cleared. This is separate from the teacher A group
target used for CLS distillation. No new expression MSE is added.

Token cross-entropy is averaged within each participating cell then across
participating cells, not diluted by unmasked cells. The inherited head sharing
policy is preserved. A separate gene center is updated from masked teacher
tokens; optional checkpointed projection/CE chunks reuse our existing iBOT
implementation. No B teacher forward is performed when disabled or no masks.

## Optional condition-filtered KoLeo

Use normalized student observed backbone CLS, not projection output. Exclude
self and same-condition candidates within the current microbatch. Average
negative log nearest-neighbor distance over eligible anchors only; no valid
anchors gives differentiable zero. This is a label-filtered project adaptation,
not unmodified unsupervised KoLeo. There is no cross-step or cross-GPU pool.

IMPORTANT: the existing runner processes one condition per forward, so this
filtered loss has NO eligible candidates there. F2/F3 are not scientifically
effective KoLeo comparisons with this sampler. A future separately audited
mixed-condition batching change is required for such experiments; do not
change DINO group targets or pair across conditions to make KoLeo nonzero.
The loss itself is tested with mixed conditions. Metrics expose eligible counts
and fractions rather than reporting the configuration switch as effectiveness.

## Logging and compatibility

Log all six raw and weighted losses, observed/masked/eligible counts and
fractions, plus existing throughput/timing/peak-memory metrics. iBOT teacher B
tokens are included in throughput accounting. Resolved loss defaults and a
pairing-contract version are included in checkpoint configuration, so old
checkpoints cannot silently resume under the new sampling/loss semantics.

Tests use tiny synthetic tensors/sparse AnnData fixtures, not formal training.

Verification: server full suite 307 passed (four existing legacy AnnData format
warnings), including four CUDA/BF16 two-update configurations and exact resumed
LoRA training with optional losses enabled. Local 168 passed/18 skipped because
optional ML dependencies are absent; unrelated user Census test excluded.
Changed Python files pass Ruff and whitespace checks; local wheel/sdist built.
