# HIM 复现范围与结构校正选择器

## 复现范围

这不是原论文全部实验的完整复现。原论文使用 Cora-ML、Power Grid、Facebook、GitHub，以及用于扩展性研究的 YouTube；常规网络覆盖 IC、LT、WIC、WLT 和多个种子预算，还包含表示学习消融、参数与效率研究。

本项目的已核验主结果是 Cora-ML / WIC / 10% 的一个固定训练种子；另完成 Ciao 真实信任拓扑上的合成 WIC 扩展，三个训练种子、每次 30 个训练级联、每个种子集合 1,000 轮独立评估。Ciao 的概率规则不同于论文 WIC 中的随机概率设置，不能称为完整复现论文实验矩阵。

2026-10-30 已补充 Cora-ML、Power Grid、Facebook 各三个独立训练种子、WIC/10%预算及每集合500次新随机流评估。新增九模型结果与本文原Ciao实验分开记录，详见 [EXPANDED_REPRODUCTION.md](EXPANDED_REPRODUCTION.md)。范围仍是部分设置复现，不覆盖全部论文矩阵。

## 贡献与创新边界

- 作者贡献：HIM 双曲表示、主动激活正则、LDO 影响力代理和 ASW 算法。
- 本项目原有贡献：本机运行适配、配置入口、模型和种子保存、缓存与随机状态修正、Ciao 适配、独立及多训练种子的对照评估。
- 本次方法扩展：在固定 HIM 表示上实现 ASW-J，结合几何注意力与共同邻居比例重新分配 ASW 惩罚。

ASW-J 是课程项目的命名，不是作者发布的方法。邻域重叠、分散选点与度折扣已有研究，本实现不主张文献首创，也不能预先称为性能改进或新架构。

## 具体定义

对已选节点 c 和当前窗口内与 c 相邻的候选 v，使用开放邻域：

`J(c,v) = |N(c) ∩ N(v)| / |N(c) ∪ N(v)|`。

原 ASW 的几何权重为 `w_g = softmax(1/d_L²(c,v))`。当候选中存在非零结构重叠时，使用

`w_new = (1-alpha) w_g + alpha J/sum(J)`。

本次实验在看结果前固定 `alpha=0.5`。保留原始 `LDO(c)/degree(c)` 的惩罚总量、窗口和邻居限制，仅改变惩罚在候选之间的分配。若所有候选 J 均为零，则沿用几何权重。实现对极小距离设数值下限，以避免相同表示造成非有限权重。原作者实现不修改；alpha=0 在普通预算下直接委托原实现，保证与原 ASW 一致。

## 验证方式

```sh
PYTHONPATH=methods/HIM methods/HIM/.venv/bin/python -m unittest discover -s methods/HIM/tests -v
methods/HIM/.venv/bin/python methods/HIM/evaluate_structural_selection.py
```

实验协议在 `results/HIM/structural_asw_protocol_2026-10-29.json` 中先冻结，随后才评估。使用三个已有 Ciao checkpoint，不重新训练或按测试结果调 alpha。与 ASW、MD、Degree、Random 在同一图、概率库、预算及 1,000 个评估随机种子下比较。重新运行原选择器并核对归档种子集合完全一致后，复用对应的旧评估结果。相同种子集合在完全相同模拟器及随机流下的结果可复用，并记录来源。

Degree 是一个确定性种子集合，在三个模型行中重复显示不代表三次独立训练。Random 的三个集合由 42、43、44 分别抽样，变异来自种子集合抽样。模拟器仍使用作者现有代码的每轮最多 100 个传播步；配对是相同初始随机流，并不保证逐边随机数完全耦合。

这是在同一既有评估流上的探索性扩展，尚不是独立确认性验证。主比较是跨三个训练模型的 ASW-J−ASW，区间使用 df=2 的 t 分布；同时报告更换节点数、选择耗时、全部比较方法和逐轮结果。无论正负都保留，不按结果挑选 alpha。

## 相关来源

- HIM 原论文与作者代码：`HIM.pdf`、`methods/HIM/seed_selection.py`；https://arxiv.org/abs/2502.13571
- Chen, Wang, Yang (2009), Efficient Influence Maximization in Social Networks：既有度折扣和减少种子冗余思路。https://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Diffusion/chen09influence.pdf
- 预先冻结的协议、逐模型结果和最终汇总见 `results/HIM/structural_asw_*2026-10-29.json`。完整训练与原评估说明见本目录 README。

## 2026-10-29 实际结果

| 方法 | 平均覆盖率 |
|---|---:|
| 原 HIM / ASW | 10.0456% |
| HIM / MD | 10.0405% |
| 本项目 ASW-J | 10.0412% |
| Degree | 9.7280% |
| Random | 8.1075% |

ASW-J 在三个模型中各更换 1、2、1 个种子，相对于原 ASW 的平均激活人数差为 -0.737、-0.716、+0.497。跨模型平均差为 **-0.318667 个节点**，95% t 区间 **[-2.073626, 1.436292]**，包含零。当前不支持性能提升，不能将该实现宣传成更优的 HIM 架构。三种 HIM 选择方式的观测均值均高于本次简单基线，但这仍是一个固定图、预算和合成传播协议下的小规模实验。

该扩展只重新分配原 ASW 惩罚，保留了原惩罚总量、窗口与邻居限制，因而选点集合变化很小。是否存在更有效的结构融合方式，需要新的事先确定的实验；本次没有按结果再调 alpha。

验证：新增 HIM 测试集 9 项通过。仓库根目录的 `unittest discover` 未发现测试，返回“NO TESTS RAN”；因此不把它作为全仓库测试通过的证据。DeepSN 自带测试属于另一套环境和方法，不属于本次修改范围。

## 2026-10-31：提交前修正与重验

修正 ASW-J 在带自环图上的归一化：总惩罚继续使用原 ASW 的 `graph.degree`，Jaccard 明确采用排除自身的开放邻域。补充自环回归与缓存指纹测试，当前26项HIM测试通过。Ciao 恢复已有结果时先重新选点，并核对协议、图、概率、checkpoint、源结果和当前集合，避免静默复用过期结果。

保留首轮源码快照及旧结果原始哈希，不把旧训练记录改写成使用修正版。新增 `verify_structural_revision.py`，重选全部12个checkpoint的五种主方法后，与原归档逐个一致；因此既有传播结果和PPT中的结论无需改变，未新增传播模拟。历史其他alpha诊断仍属于当时源码版本，不据此调参。完整记录见 `results/HIM/structural_revision_verification_2026-10-31.json`。

核验顺序（从仓库根目录）：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 methods/HIM/.venv/bin/python methods/HIM/verify_structural_revision.py
PYTHONPATH=methods/HIM methods/HIM/.venv/bin/python -m unittest discover -s methods/HIM/tests -v
OMP_NUM_THREADS=1 methods/HIM/.venv/bin/python methods/HIM/verify_expanded_reproduction.py
OMP_NUM_THREADS=1 methods/HIM/.venv/bin/python methods/HIM/verify_coverage_selection.py
```

历史结果恢复必须具有与当前选择器及逐模型文件精确匹配的修订审计；任意其他源码变化会拒绝复用。重新训练应另设实验协议与结果目录，模型和缓存不随Git分发。
