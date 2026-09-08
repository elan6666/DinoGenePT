# 768-wide, reduced-depth 50k capacity test (2026-09-09)

The user requires preserving the 768-dimensional gene/backbone representation.
The previous width16 / Student495570 tiny recipe is a historical optional
variant, not the current 50k resolver default. Its global batch256 capacity does
not transfer to this model. Existing checkpoints remain untouched.

## Configuration

- Native backbone: width768, depth4, three unidirectional KDA layers + one MLA.
- KDA: 6 heads x128; MLA: 12 heads x64, shared32, query rank192, KV rank128.
- Expression basis256, FFN expansion1, checkpointing enabled, chunk16.
- Shared DINO/iBOT head: hidden1024, bottleneck128, prototypes8192.
- Student **40,639,827** parameters, including backbone **36,434,387**.
  The EMA Teacher is a separate copy, not included in the Student count.
- Versus the previous width768/depth12 Student82,852,215, depth and projection
  dimensions are reduced; gene/backbone width and gene vocabulary are preserved.
- Current masking, five losses, EMA and formal LR scheduling are unchanged.

New 50k config resolution selects `genecompass50k_balanced_recipe.json`.
Its microbatch4 is a conservative placeholder, not a formal training launch.
The capacity script explicitly overrides batch. No automatic batch256 reuse.

## Test protocol

Two RTX5090 GPUs; DDP; accumulation1; bf16 autocast with fp32 parameters.
Distinct real GeneCompass50k cells per rank, each with 2048 valid genes.
Maximum crop lengths: 2048,2048,820,820. Exactly half the global views are masked,
concentrated in global0 as a legal allocation worst case; target/mask ratio20%.
Each update includes all losses, backward, gradient clipping, AdamW and EMA.
Loss and gradient finiteness are checked on both ranks. Memory is the maximum
over ranks. The probe uses fixed LR1e-6, not the formal LR schedule, and repeats
the selected batch: this measures capacity, not convergence or loader throughput.

| Per GPU | Global | Result |
|---:|---:|---|
|12|24|3 full updates passed|
|16|32|10 full updates passed; recommended capacity candidate|
|19|38|10 full updates passed|
|20|40|First update passed; second update CUDA OOM on both ranks|

At19/rank, peak allocated29,905,211,392 bytes, reserved31,874,613,248 bytes;
last-four-step throughput approximately11.9–12.2 cells/s. The tested boundary
is19/rank under this exact allocation/configuration, not a hardware-independent
maximum or a guarantee that a full epoch is stable.

At16/rank, peak allocated25,334,491,648 bytes (23.59GiB), reserved29,246,881,792
bytes (27.24GiB); last-three-step throughput11.69–11.98 cells/s. Recommend this
global32 candidate for headroom instead of operating at the global38 boundary.
This is a recommendation only; no full training run was launched.

## Server evidence

Root: `/data/yilangliu/DinoGenePT/results/capacity/20260909-balanced-b{12,16,19,20}`.
Successful runs contain `completion.json`; failure evidence is in
`.runtime/capacity-balanced-b20.log`. Raw data, logs and weights stay server-side.

- Data manifest SHA256: `73d490fc0e8fbfdb60f3827000d582daca15487ec80dc2438a1f53d3454d0d27`.
- Data config SHA256: `945ad7f7036fdb0032b90b2dcb6de9275200b84ac96dba8044aadbdf8aae0768`.
- Balanced recipe SHA256: `c9d85a6bfc016dfdac2d7539003653cc9857582ea9b2fb611c01646ca1061c76`.
- Probe source SHA256: `7373fbc804031fcfe2a729daf61e19be9fa78b5aac70ca19e5b11d7c7d050f46`.

No formal pretraining, fine-tuning or monitoring is started by this test.
Server CPU validation:24 tests passed (balanced/tiny recipes, pretraining system,
GeneCompass launcher). Scoped local Ruff and diff whitespace checks passed.
