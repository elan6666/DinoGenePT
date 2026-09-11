# H组训练机制与E组显存优化消融

2026-09-12。状态：已选择的实验设计，未实现新开关、未启动训练。
仅研究Dense模型；不包含MoE、专家均衡或生成式后训练。

## 共同协议

- 使用同一版本数据、词表、split、骨干、初始化seed、crop、batch和训练预算。
- 新词表须先迁移训练token并完成哈希/覆盖校验，不能按旧编号直接换表。
- 不搜索现有五项loss权重；EMA沿用当前余弦机制，独立EMA消融仍保留。
- 初筛固定batch108；若该比较组无法全部运行，整组采用相同安全batch。
- 模型选择仅用validation。各方法获得相同LR搜索预算，最终重要结果多seed复验。
- 不改动LoRA微调优化器；先隔离预训练机制，微调协议保持一致。
- 以下是原生小模型适配，不是三家完整训练配方复现。正式实现先审计固定版本源码。

## H-LR：三家学习率机制

p为成功优化器更新数/总更新数。所有比例以step定义，不按crop数计。

| ID | 适配调度 | 来源与改动 |
|---|---|---|
| H-LR0 | 16%线性warmup，随后余弦到1e-6 | 当前基线 |
| H-LR-G | 5%线性warmup，随后余弦到峰值的20% | GLM-5预训练形状；5%为本项目设定；不混入其mid-training阶段 |
| H-LR-K | 1%线性warmup，随后余弦到1e-6 | Kimi-K3的1% warmup与余弦；最低值为项目设定 |
| H-LR-D | 5%线性warmup，5%-62%恒定，62%-89%余弦到峰值10%，89%-100%保持 | DS4.1的恒定/衰减/收尾形状；warmup改为5%，token阶段边界近似映射到step比例 |

warmup从0开始；衰减段采用
`lr_min + (lr_peak-lr_min)*(1+cos(pi*u))/2`，u为该衰减段进度。
这是固定预算配方比较，warmup与末值也属于配方变量，不能归因为单一曲线形状。
另设受控对照：GLM/Kimi余弦与DS恒定-余弦-收尾共用5% warmup、1e-6末值。
GLM/Kimi在该控制下若完全相同，只运行一次。

第一轮全部使用当前AdamW，基准LR候选1e-4、2e-4、4e-4，继续使用
`peak = base * sqrt(global_batch/1024)`。该缩放是项目协议，不声称来自三家。
batch108时峰值约3.25e-5、6.50e-5、1.30e-4。不做周期重启。
既有EMA初值0.990/0.994/0.998、batch、温度等H组候选保留，分轮执行。

## H-OPT：Muon方式

先固定一个LR调度，不把三种LR与三种优化器全部捆绑。

共同保守参数组：gene embedding用AdamW且decay=0；表达编码器、
DINO/iBOT投影和原型头、Norm、bias、门控/衰减标量先保持AdamW。
Muon只用于明确标记的骨干Linear矩阵，不按ndim==2自动分配。
每个参数必须且只能归属一个组；共享DINO/iBOT头不得重复注册。

| ID | 骨干矩阵更新 | 独立变化 |
|---|---|---|
| H-O0 | 当前AdamW | 基线 |
| H-O1 | 普通整矩阵Muon | 通用桥接对照 |
| H-O-G | GLM Muon Split | 对具有独立Head布局的Q/K/V上投影逐Head处理；共享低秩投影不切分 |
| H-O-K | Kimi-K3 Per-Head Muon | Q/K/V逐Head处理，加其权重裁剪机制的适配；裁剪只作用于经过推导的Softmax注意力投影 |
| H-O-D | DS4.1 Head-wise Muon | Q/K逐Head处理，其他骨干矩阵Muon；第一轮保留共同输入/输出参数组 |

H-O-G与不带裁剪的H-O-K若经源码审计后等价，复用同一运行，不能凭品牌名称当成不同算法。
为隔离裁剪贡献，对H-O-K设clip off/on；不将QK裁剪机械用于KDA状态衰减或DINO原型logit。
DS4.1不同于Kimi的Q/K/V划分，应通过模块布局清单确认Q/K边界，不能切开共享语义。

Muon动量候选默认0.95。正式实现须固定Newton-Schulz迭代、矩阵方向、
零梯度处理和更新RMS缩放；DS适配使用其0.18缩放，GLM/Kimi按各自来源核对。
不能假设不同Muon实现共用一个数值LR即可公平比较。每种方法使用相同次数的
LR搜索，并记录实际每组LR、更新RMS、||delta W||/(||W||+eps)、各Head梯度尺度。
第一轮decay沿用项目分组值，明确不等同于官方全配方。

### H-O-D-S：DS4.1词表更新扩展

在H-O-D之外单独比较gene embedding的AdamW与Momentum+Sinkhorn-balanced更新。
借鉴DS的动量0.95、11次交替归一化、近零行阈值1e-3、更新缩放0.18；
epsilon按实际状态dtype核验，不在低精度中盲用1e-20。词表不施加decay。
不同时改变DINO原型头：它不是语言模型prediction head。
必须验证PAD行、从未出现行、历史出现但当前缺席行和稀有基因；历史动量
仍可能更新缺席行，不能声称等价于逐行冻结。不满足身份/数值门槛则不启用。

LR与优化器分别筛选后，仅对少量入选组合交叉验证；不预先指定同品牌绑定。
记录optimizer step耗时、裁剪触发率、状态显存、下游效果。Teacher EMA在
所有Student更新及权重裁剪完成后更新一次；AMP跳步不推进EMA或调度。

## E-MEM：分块计算iBOT投影与loss

这是工程优化测试，不属于改变学习机制的H组。当前已只投影被遮挡位置，
新实验是在这些位置上进一步分块，不改变遮挡、原型数或KoLeo池。

| ID | iBOT被遮挡token处理块大小 |
|---|---|
| E0 | 当前不分块实现 |
| E256 | 256 |
| E512 | 512 |
| E1024 | 1024 |

Teacher和Student均按对应基因顺序分块。共享投影头仍共享；Teacher采用no_grad，
中心向量对整个logical batch冻结，所有块的sum/count聚合后只更新一次。
Student必须通过重计算/自定义autograd或等价的正确释放策略降低保存激活量。
仅在Python循环中累加带计算图loss，不构成已证明的显存优化。

必须保持按细胞/视角的原始权重；不改成按块平均、不按token数改变细胞权重。
共享头的CLS分支梯度与基因分支梯度须正确合并；DDP不能每块重复all-reduce，
也不能漏同步。梯度裁剪、优化器、EMA均每个逻辑step一次。

### 正确性门槛

- 相同权重、输入和mask，比较FP32 loss及所有参数梯度：初始容差atol=1e-6、rtol=1e-4。
- BF16比较使用独立精度测试，初始容差atol=1e-3、rtol=1e-2；同时报告最大误差及梯度相对L2误差。
  这些是待校验阈值，不能为了通过而静默放宽；必须和未优化实现的精度误差对照。
- 覆盖零遮挡、不完整尾块、不同细胞mask数、PAD、共享头、双卡及断点恢复。
- 比较一个完整更新后的Student、Teacher、center与优化器状态，确认语义不变。

### 性能门槛

先测固定batch108，再单独测试容量，不能用更大batch替代同batch速度比较。
GPU不可与无关任务争抢；若无法独占测量须标记受干扰，不下速度结论。
固定软件、精度和checkpoint策略；每组10步预热+50步测量，3次重复，轮换顺序。
CUDA同步计时、重置峰值统计；OOM如实记录，不隐藏为成功。
报告max_memory_allocated/reserved、step中位数/P95、cells/s、optimizer耗时，
以及不同块大小的速度-显存折中。
初步采用标准：通过正确性，峰值allocated降低至少10%且step中位数变慢不超过5%；
不达标可作为容量模式保留，但不能称为默认加速。最后短轨迹回归比较各项loss漂移。

## 执行边界及来源

目前只登记设计，不改运行配置、不启动训练。先做E组等价性/性能测试，
再用验证过的统一实现进行H组，以免不同实验使用不同工程路径。

- [GLM-5 §2.1、§2.4.1、附录A](https://arxiv.org/html/2602.15763v1)：Muon Split、分块输出、预训练LR；不称为GLM5.3完整训练配方。
- [Kimi-K3 §2.5、§3.2–3.3](https://arxiv.org/html/2607.24653v1)：Per-Head Muon、裁剪和1%warmup/余弦。
- [DS4.1 §2.5、§4.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)：Q/K Head-wise Muon、Sinkhorn更新及恒定/衰减/收尾。
