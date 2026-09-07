# Default CUDA capacity and numerical audit

These are engineering probes, **not pretraining epochs, downstream scores or
ablations**. Real training remains zero epochs. No other research job was stopped.

All capacity probes used a hash-checked completed raw Census shard, full60,664
vocabulary,12/768 default backbone,119,039,863 Student parameters, full heads,
two clean Teacher globals, four Student views, five losses, backward, AdamW and
EMA. Inputs stress upper crop bounds: globals2048/2048, locals820/820. FP32
weights/state with BF16 autocast; RTX5090, torch2.13.0+cu130. Two steps only.

| Probe | Microbatch/rank | Ranks | Second step | Total cells/s | Peak allocated/max rank |
|---|---:|---:|---:|---:|---:|
| Original native chunk | 2 | 1 | 10.094s | 0.198 | 12,409,045,504 bytes |
| Original native chunk | 4 | 2 | 10.752s | 0.744 | 22,912,157,696 bytes |
| Candidate parallel chunk | 4 | 2 | 3.288s | 2.433 | 23,121,669,632 bytes |

The3.27x candidate speed ratio is a single warmed step comparison, not a repeated
statistical benchmark or expected whole-training speedup. Allocation is not total
driver memory. Real fractional crops differ from this worst-bound probe. Losses
and gradients were finite but high from-scratch reconstruction gradients mean
two optimizer steps are not evidence of convergence or stable long training.

## Candidate is not promoted

`parallel_chunk_kda` algebraically factors within-chunk updates into a value
term minus a state-read term, batching independent triangular solves while
preserving sequential boundary-state updates. It is native PyTorch scheduling,
not an upstream fused-kernel import or an attention architecture ablation.

CPU recurrence/gradient, read-only-token and strong-decay tests pass. CUDA
2049-token/6-head/128-width original/candidate values and gradients pass
atol2e-5/rtol2e-4. Whole12/768 backbone FP32 errors are small: CLS relativeL2
6.54e-7 and gene output1.47e-6. This does **not** establish BF16 equivalence.

The predeclared BF16 elementwise gate failed on11/393,216 gene outputs, max
absolute difference0.09375. CLS/gene relativeL2 differences are1.35%/1.59%.
For context only, original BF16-vs-FP32 relative errors were4.36%/3.11%, versus
candidate4.37%/3.11%. These diagnostics do not override the failed gate. Retain
`kda_implementation=chunk` as default until precision/order preservation is
resolved; do not merely widen tolerances to accept the faster candidate.

## Evidence on the server

All paths below are relative to `/data/yilangliu/DinoGenePT/results/`.

| Receipt | SHA256 |
|---|---|
| capacity-single-default-v1/completion.json | d76cad37bd99708468e9c2f05ebf394cf194d6a41fbb54b44d2446569f3a8236 |
| capacity-ddp-default-v1/completion.json | 94d4d1e0ff272a417e004d25dc819445c42e14acd89d46322c7bf8ccf8ef7d33 |
| capacity-ddp-parallel-default-v3/completion.json | 33d0051364ef2bb0e8a76b7a4b0ee7ee7df567b590e5ef39399207fd41f517c3 |
| kda-parallel-cuda-numerics-v2.json (failed) | 46d172ac7a4aee7cc7191d98f8934ce802c3d07dda62c777ab4ddd6bddb6a2e5 |

Candidate DDP attempt v1 stopped on a false-positive peer ownership check:
Elastic uses separate rank sessions. Verified common-torchrun-parent handling
fixes this, with tests rejecting arbitrary-shell siblings. Attempt v2 correctly
refused a separate live PID688199; it disappeared before ownership attribution,
so no project attribution is claimed. V3 ran only after free-GPU revalidation.
Later PID775988 occupied GPU0 from `/home/yilangliu` with system Python; the
numerical audit bound only free GPU1. Do not bypass the guard to reserve both.

Use `scripts/probe_pretraining_capacity.py` and
`scripts/audit_kda_optimization.py` to reproduce with fresh output paths. These
probes cannot emit a formal training checkpoint or certify a partial corpus.
