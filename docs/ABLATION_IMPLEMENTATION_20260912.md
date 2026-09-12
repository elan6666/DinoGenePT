# 消融实现与微型验收（2026-09-12）

本轮完成代码/配置及短测，不是正式消融结果。没有启动完整epoch、API语料生成、
GraD-Pert或定时监控。当前默认recipe没有改为Muon、HVG或iBOT分块。

## 实现入口与覆盖

`src/dinogenept/cell/ablations.py` 是唯一消融注册表：67个配置条目，
其中60个预训练条目、7个扰动知识条目。包含重复基线/受控对照，
不能把67个条目说成67种独立算法。单模型、单训练入口、同一评估器。

| 组 | 已接入原生代码的变化 |
|---|---|
| H | DINOv2/GLM5/KimiK3/DS4.1适配调度；LR峰值、EMA初值、温度、batch配置 |
| H-OPT | AdamW、整矩阵Muon、Q/K/V分Head、DS的Q/K分Head；Kimi MLA QK-Clip；独立词表Sinkhorn |
| S | 深度4/8/12，维度256/512/768，FFN扩张1/4（没有2） |
| C | KDA/全MLA/CellFM retention，AttnRes/标准残差/DeepNorm形式，单双向KDA、短卷积0/4、SiTU/SwiGLU/SGLU |
| C-迁移 | 标量Gated Delta、GQA；Qwen-inspired与GLM-dense-inspired组合。不是官方完整模型或预训练权重迁移 |
| V | Local数量2/4/8，连续表达+均匀/加权候选抽样，Global比例，固定HVG-K，独立/共享iBOT头 |
| K | TextBase/CellGene主锚点，GO/Protein/Pathway/HPA单源及关闭知识Local；缺失源仍不补零 |
| E | iBOT分块0/256/512/1024，骨干checkpoint开关，三种既有KDA执行路径 |

没有加入MoE，也没有改五项loss权重。扰动任务沿用已有LoRA及共同评估，
H组先只改变预训练，避免同时修改微调优化器造成混淆。

服务器仓库中运行：

```bash
.venv/bin/python scripts/check_ablation_suite.py --list
.venv/bin/python scripts/check_ablation_suite.py \
  --recipe configs/cell/genecompass50k_width256_recipe.json --resolve H-O-G
.venv/bin/python scripts/check_ablation_suite.py --smoke H-O-G E-IBOT-256 \
  --device cuda --output results/new-smoke-receipt.json
```

resolve输出JSON不启动任务；保留输入数据身份，若基础配置含output，
自动使用其`ablations/<ID>`子目录。新receipt必须使用未占用路径。
不同数据集传各自基础配置，不复制训练器。正式HVG配置必须提供训练集专用的
冻结ID清单、SHA256及对应数据manifest身份；没有清单直接拒绝，不猜HVG。

## 优化器/梯度安全

- 原生Muon基于审计的NS/Nesterov计算；同形状Head批量NS，但各Head独立归一化。
- 精确处理MLA交错K/V行；DS的V按整矩阵处理，不误切成Q/K/V全部分Head。
- gene embedding保持AdamW、decay=0；Sinkhorn只有H-O-D-S显式开启。
- Sinkhorn按DS Algorithm1：近零Nesterov行屏蔽、11次交替L2归一化，
  最后乘sqrt(列数)与0.18。PAD和从未出现行不变；历史动量可更新当前缺席行。
- 主干Linear使用Muon，输入编码、DINO/iBOT头、Norm/bias/标量保留AdamW。
- FP32主参数、FP32 NS；BF16只用于autocast。此数值路径和小模型分组属于本项目适配。
- Kimi裁剪仅对MLA有效；跨rank及所有视角累计max，optimizer后、Teacher EMA前处理一次。
  私有Q/K乘sqrt(gamma)，共享通道的Q乘gamma，K共享通道不动。GQA组合明确拒绝此裁剪。
- Muon断点恢复、权重更新、零梯度及身份行保护均有测试。

## iBOT实现与测量

Student分块checkpoint的是“投影+CE”，Teacher目标也在同一块重计算，
避免保留全部N×原型数矩阵。所有块共享冻结center，累计sum/count后统一更新。
保留每细胞/视角权重；零遮挡块依然连接独立头的零梯度；DDP同步不在块循环内。

5090，PyTorch2.13.0+cu130，BF16，2048个遮挡token、width256、
投影隐藏1024/瓶颈128/8192原型。每组2步预热+5步测量，3轮轮换顺序。
下表耗时为三轮各自中位数再取中位数，包含head前向+反向；**不是完整骨干**。

| chunk | 峰值allocated（MiB） | 相对减少 | 耗时（ms） |
|---|---:|---:|---:|
| 不分块 | 430.02 | — | 2.55 |
| 256 | 143.01 | 66.7% | 27.76 |
| 512 | 185.26 | 56.9% | 15.43 |
| 1024 | 269.77 | 37.3% | 6.94 |

结论：实现可作为省显存选项，但此局部短测明显变慢，不符合“默认加速”的门槛，
不默认开启。不能将这里的百分比外推到整模型或batch108；尚未执行设计中
10+50步的正式全模型性能测量，也未重复最大batch压力测试。

## 验收结果与边界

- 服务器CPU：287 passed、3 CUDA-only skipped；随后这3个BF16梯度测试在GPU通过。
- 双5090：60个预训练配置条目各两次更新，检查梯度有限、Student跨rank参数逐位一致。
  E组补测真正小块和rank1完全零遮挡；Muon合批优化后6个相关条目再次双卡通过。
- 烟雾测试是96基因、每卡2细胞、短序列、小骨干；层数/宽度/块大小的物理测试配置
  全部写进receipt。**不是正式参数量、batch32/64/108的容量证明，也不是效果比较。**
- K组主锚点/源选择/缺失源测试在原生扰动模型中完成；不拿预训练烟雾代替微调验证。
- FP32比较loss及梯度；完整Student/Teacher/center更新回归通过。
  BF16额外比较loss、center统计与整体梯度相对L2误差<1%。
- 本地轻量测试167 passed、18 optional-dependency skipped；未纳入用户原有未提交
  `test_census_query_groups.py`（本地缺pandas，直接全收集会报错）。未修改该文件。
- 本地/服务器Ruff、wheel+sdist、CLI检查通过。数据、旧checkpoint、默认训练recipe未改。
- 正式数据训练前仍需完成新词表迁移、HVG来源和真实知识向量版本审计。语料纯度、
  无泄露泛化及下游提升不能由本次synthetic测试证明。

原始JSON留在服务器`/data/yilangliu/DinoGenePT/results/`；摘要与哈希见
`docs/results/ABLATION_SMOKE_20260912.json`。历史设计文件是研究来源，不覆盖本验收状态。
