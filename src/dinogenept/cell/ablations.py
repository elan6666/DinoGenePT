"""One-factor config overlays; one native model and one evaluation contract.

Registry entries never authorize training. Resolve a dataset config separately;
all lineage fields remain intact. None explicitly unsets a inherited LR option.
"""

from copy import deepcopy
from pathlib import Path

from .backbone import BackboneConfig
from .pretraining import HeadConfig
from .sampling import CropConfig


def ablation_registry():
    rows = {"BASE": {}}
    rows["H-LR0"] = {"training": dict(lr_scheduler="dinov2_step_cosine", lr_warmup_fraction=.16,
                                     min_learning_rate=1e-6, optimizer="adamw")}
    for label, scheduler in (("G", "glm5_cosine"), ("K", "kimi3_cosine"), ("D", "ds41_plateau_cosine")):
        rows[f"H-LR-{label}"] = {"training": dict(lr_scheduler=scheduler, lr_warmup_fraction=None,
                                                 min_learning_rate=None, optimizer="adamw")}
    for value in (1e-4, 2e-4, 4e-4):
        rows[f"H-LR-peak-{value:g}"] = {"training": {"learning_rate": value}}
    for value in (.990, .994, .998):
        rows[f"H-EMA-{value}"] = {"heads": {"ema_initial": value}}
    for value in (.07, .1, .2):
        rows[f"H-TEMP-{value}"] = {"heads": {"student_temperature": value}}
    for value in (32, 64, 108):
        rows[f"H-BATCH-{value}"] = {"training": dict(microbatch=value // 2, world_size=2, accumulation=1)}
    for label, mode in (("0", "adamw"), ("1", "muon"), ("G", "glm5_muon"),
                        ("K", "kimi3_muon"), ("D", "ds41_muon")):
        rows[f"H-O-{label}"] = {"training": dict(optimizer=mode, qk_clip_threshold=None,
                                                embedding_optimizer="adamw")}
    rows["H-O-K-clip"] = {"training": dict(optimizer="kimi3_muon", qk_clip_threshold=100.)}
    rows["H-O-D-S"] = {"training": dict(optimizer="ds41_muon", embedding_optimizer="sinkhorn")}
    for value in (4, 8, 12):
        rows[f"S-DEPTH-{value}"] = {"backbone": {"depth": value}}
    for value in (256, 512, 768):
        rows[f"S-WIDTH-{value}"] = {"backbone": dict(width=value, kda_heads=value // 128,
                                                    mla_heads=value // 64)}
    for value in (1, 4):
        rows[f"S-FFN-{value}"] = {"backbone": {"ffn_expansion": value}}
    for value in (2, 4, 8):
        rows[f"V-LOCAL-{value}"] = {"crops": {"local_count": value}}
    rows["V-UNIFORM-CAP"] = {"crops": {"cap_sampling": "uniform"}}
    rows["V-HVG-K128"] = {"crops": dict(local_sampling="hvg", local_hvg_k=128)}
    rows["V-HVG-K256"] = {"crops": dict(local_sampling="hvg", local_hvg_k=256)}
    rows["V-GLOBAL-FIXED40"] = {"crops": {"global_scale": [.4, .4]}}
    rows["V-GLOBAL-FIXED100"] = {"crops": {"global_scale": [1., 1.]}}
    rows["V-INDEPENDENT-HEAD"] = {"heads": {"ibot_separate_head": True}}
    rows["C-FULL-MLA"] = {"backbone": {"mixer_type": "full_mla"}}
    rows["C-ERET-MIXER"] = {"backbone": {"mixer_type": "eretnet"}}
    rows["C-CELLFM-BLOCK"] = {"backbone": dict(mixer_type="eretnet", residual_type="deepnorm", ffn_type="sglu")}
    rows["C-STANDARD-RESIDUAL"] = {"backbone": {"residual_type": "standard"}}
    rows["C-KDA-BIDIRECTIONAL"] = {"backbone": {"kda_direction": "bidirectional"}}
    rows["C-KDA-CONV4"] = {"backbone": {"short_convolution": 4}}
    rows["C-SWIGLU"] = {"backbone": {"ffn_type": "swiglu"}}
    rows["C-GATED-DELTA"] = {"backbone": {"delta_rule": "scalar"}}
    rows["C-GQA"] = {"backbone": {"global_attention": "gqa"}}
    rows["C-QWEN-INSPIRED"] = {"backbone": dict(delta_rule="scalar", global_attention="gqa",
                                               residual_type="standard", ffn_type="swiglu")}
    rows["C-GLM-DENSE-INSPIRED"] = {"backbone": dict(mixer_type="full_mla", residual_type="standard",
                                                    ffn_type="swiglu")}
    for value in (0, 256, 512, 1024):
        rows[f"E-IBOT-{value}"] = {"heads": {"ibot_chunk_size": value}}
    rows["E-CHECKPOINT-OFF"] = {"backbone": {"gradient_checkpointing": False}}
    for value in ("chunk", "batched_chunk", "parallel_chunk"):
        rows[f"E-KDA-{value}"] = {"backbone": {"kda_implementation": value}}
    for main in ("TextBase", "CellGene"):
        rows[f"K-ANCHOR-{main}"] = {"perturbation": {"main_anchor": main}}
    for name in ("GO", "Protein", "Pathway", "HPA"):
        rows[f"K-ONLY-{name}"] = {"perturbation": {"local_sources": [name]}}
    rows["K-NO-LOCAL"] = {"perturbation": {"local_sources": []}}
    return rows


def resolve_ablation(base, name):
    variants = ablation_registry()
    if name not in variants:
        raise ValueError(f"Unknown ablation {name}; no silent fallback")
    result = deepcopy(base)
    for section, changes in variants[name].items():
        result.setdefault(section, {}).update(changes)
    BackboneConfig(**result["backbone"])
    HeadConfig(**result["heads"])
    CropConfig(**result["crops"])
    result["ablation"] = {"id": name, "overlay": deepcopy(variants[name]), "schema": 1}
    if "output" in result:
        result["output"] = str(Path(base["output"]) / "ablations" / name)
    return result
