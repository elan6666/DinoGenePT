"""Isolated real-data, maximum-crop DDP capacity test; never saves model weights."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dinogenept.cell.backbone import BackboneConfig
from dinogenept.cell.genecompass_data import GeneCompassDataset
from dinogenept.cell.pretraining import HeadConfig, PretrainingSystem
from dinogenept.cell.sampling import CropConfig, collate_crops
from dinogenept.cell.train import _gpu_guard, _move
from dinogenept.provenance import atomic_write_json, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--recipe', type=Path, default=Path('configs/cell/census500k_default_recipe.json'))
    args = parser.parse_args()
    if args.microbatch < 2 or args.steps < 2:
        raise ValueError('Need at least two cells and two complete optimizer steps')
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    _gpu_guard()
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl')
    cfg = json.loads(args.config.read_text())
    recipe = json.loads(args.recipe.read_text())
    cfg['backbone'] = {**recipe['backbone'], 'genes': cfg['backbone']['genes']}
    cfg['heads'], cfg['crops'], cfg['training'] = recipe['heads'], recipe['crops'], recipe['training']
    crops = CropConfig(**{**cfg['crops'], 'global_scale': (1., 1.), 'local_scale': (.4, .4)})
    data = GeneCompassDataset(cfg['data_manifest'], crops)
    manifest_hash = data.verify()
    if manifest_hash != cfg['data_manifest_sha256']:
        raise ValueError('Manifest differs from training config')
    selected, offset = [], 0
    for index, shard in enumerate(data.shards):
        ids, _ = data._read(index)
        selected.extend((offset + np.flatnonzero((ids > 0).sum(axis=1) == crops.cap)).tolist())
        offset += shard['cells']
        if len(selected) >= args.microbatch * world:
            break
    if len(selected) < args.microbatch * world:
        raise ValueError('Not enough distinct maximum-length real cells')
    rows = [data[i] for i in selected[:args.microbatch * world][rank::world]]
    batch = collate_crops(rows)
    # Legal worst-case allocation: exactly half of 2B globals masked, all in
    # global0. This maximizes the shared prototype-head allocation per call.
    for i, row in enumerate(rows):
        for key in ('targets', 'hidden'):
            batch['views'][0][key][i] = torch.from_numpy(row['views'][0][key])
            batch['views'][1][key][i].zero_()
    batch = _move(batch, device)
    torch.manual_seed(cfg['training']['seed'])
    model = PretrainingSystem(BackboneConfig(**cfg['backbone']), HeadConfig(**cfg['heads'])).to(device).train()
    wrapped = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    model.teacher.load_state_dict(model.student.state_dict())
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-6,
                                 betas=tuple(cfg['training']['betas']),
                                 weight_decay=cfg['training']['weight_decay'])
    identity = dict(purpose='capacity_only_not_training', microbatch=args.microbatch,
                    world_size=world, accumulation=1, effective_batch=args.microbatch * world,
                    config_sha256=digest_file(args.config), recipe_sha256=digest_file(args.recipe),
                    mask_layout='worst_case_all_global0_half_batch', manifest_sha256=manifest_hash,
                    backbone=cfg['backbone'], heads=cfg['heads'], crops=asdict(crops),
                    view_lengths=[v['gene_ids'].shape[1] for v in batch['views']],
                    precision='bf16_autocast_fp32_parameters', gpu=torch.cuda.get_device_name(),
                    source_sha256=digest_file(Path(__file__)))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(args.output / 'started.json', identity)
        print(json.dumps(identity), flush=True)
    records = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = wrapped(batch, step=50 + step, total_steps=100)
        if not all(bool(torch.isfinite(v)) for v in losses.values()):
            raise FloatingPointError('Nonfinite loss')
        losses['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.update_ema(51 + step, 100)
        torch.cuda.synchronize()
        stats = torch.tensor([time.monotonic() - start, torch.cuda.max_memory_allocated(),
                              torch.cuda.max_memory_reserved()], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        seconds, allocated, reserved = stats.tolist()
        records.append(dict(step=step + 1, seconds=seconds, peak_allocated_bytes=int(allocated),
                            peak_reserved_bytes=int(reserved), cells_per_second=args.microbatch * world / seconds,
                            gradient_norm_rank0=float(norm), loss_rank0=float(losses['total'].detach())))
        if rank == 0:
            print(json.dumps(records[-1]), flush=True)
            atomic_write_json(args.output / 'progress.json', dict(identity=identity, steps=records))
    if rank == 0:
        atomic_write_json(args.output / 'completion.json', dict(status='passed', identity=identity, steps=records))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
