# HIM：多网络复现、结构校正实验与汇报说明

## 完成范围

本轮新增三个原论文网络（Cora-ML、Power Grid、Facebook），每个网络使用 142、143、144 三个独立训练种子，共 9 次新训练。均采用 WIC、10% 预算、30 个训练级联、最多 200 epoch、每 epoch 100 次在线验证、patience=10。测试使用新的随机种子基数 100042，每个种子集合 500 次模拟。保留 Ciao 上三个已有模型、5% 预算及每集合 1,000 次模拟的扩展实验。

这属于作者代码与固定概率库下的部分设置复现及课程扩展。没有完成 GitHub/YouTube、全部 IC/LT/WIC/WLT、多预算训练、表示学习消融或全部论文基线。MD 只消融选择器，不能代替激活正则、双曲空间等表示学习消融。

## 模型、数据与我们自己的工作

- 数据集是社交图。HIM 是表示学习与选点方法，WIC 是传播机制，ASW/MD/ASW-J 是固定表示上的选择器。Degree/Random 是不用 HIM 训练的简单选点基线。
- 原作者贡献包括双曲表示、社交与激活关系联合学习、主动激活正则、LDO 代理以及 ASW。
- 本项目补充本机适配、配置入口、模型/种子保存、缓存隔离、随机状态控制、独立评估、Ciao 适配和多网络多训练种子的对照。
- ASW-J 是已实现的课程方法扩展：只重新分配 ASW 惩罚，不改变 HIM 表示、损失、候选窗口或惩罚总量。不主张邻域折扣思路的文献首创。

## 新复现实验结果

表中为各网络三次训练的平均覆盖率。网络间概率分布、拓扑不同，不对这些覆盖率作跨网络排名。

|网络|ASW|MD|ASW-J|Degree|Random|
|---|---:|---:|---:|---:|---:|
|Cora-ML|38.1440%|37.6479%|38.1832%|34.3678%|37.8651%|
|Power Grid|18.7217%|18.2440%|18.7218%|18.2193%|15.2432%|
|Facebook|31.8099%|31.8031%|31.8312%|26.5291%|31.9168%|

|网络|ASW-J−ASW 平均激活差|跨3次训练95% t区间|更换种子数|
|---|---:|---|---|
|Cora-ML|+1.102|[-0.754, 2.958]|[2, 2, 1]|
|Power Grid|+0.008|[-1.707, 1.723]|[6, 10, 8]|
|Facebook|+1.263|[-2.237, 4.762]|[1, 1, 3]|
|Ciao（原扩展设置）|-0.319|[-2.074, 1.436]|[1, 2, 1]|

区间针对固定图、概率库和三次训练下的配对平均差。每集合多次模拟不等于多次训练。Degree 每网络只有一个确定性集合，其重复行只便于对齐；Random 的变化来自三个抽样集合。我们没有对网络合并求一个总均值，也未对多项探索性对照进行多重比较校正。

本轮三个论文网络与 Ciao 的 ASW-J−ASW 区间全部包含零，没有观察到稳定正向增益。Facebook 上 Random 的观测平均覆盖率略高于 HIM；样本仍小，不能反过来宣称 Random 普遍更优。

## 与论文表格的区别

本地论文第7页定义 WIC 的概率范围为 [0,0.2]。本轮保留作者附带的概率库：Cora-ML 与 Power Grid 实际约 [0.1,0.2]，Facebook 实际约 [0,0.1]。此外原论文第9页 Table IV 的模拟轮数、训练选择与本轮不完全一致。因此以下表只说明数值背景，不能据此说复现了论文表格、达到/超过论文或作算法排名。

|网络|论文 Table IV HIM，WIC/10%|本轮作者代码/固定概率库|
|---|---:|---:|
|Cora-ML|38.87%|38.1440%|
|Power Grid|19.13%|18.7217%|
|Facebook|32.69%|31.8099%|

## ASW-J 与思考

对当前种子 c 和窗口内相邻候选 v，使用开放邻域 Jaccard：J=|N(c)∩N(v)|/|N(c)∪N(v)|。将原几何 softmax 注意力与 J 的归一化值各混合一半；没有结构证据时沿用几何权重。alpha=.5 在评估前固定，不按结果调参。其余 alpha=[0,.25,.5,.75,1] 只诊断种子集合的变化，不评估它们的传播结果、也不选择最优 alpha。

这次验证可检验“共同邻居是否能辅助几何折扣”的想法，但不能预设它会提高覆盖率。种子集合变化与结果的关系只是观察。原候选窗口和惩罚总量可能限制修改幅度，共同邻居也不等于真实的传播可达或累计覆盖；要区分这些解释，仍需各自的额外消融，不能写成已证明的机制。

## 运行与核验

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 methods/HIM/.venv/bin/python methods/HIM/expanded_reproduction.py
PYTHONPATH=methods/HIM methods/HIM/.venv/bin/python -m unittest discover -s methods/HIM/tests -v
methods/HIM/.venv/bin/python methods/HIM/verify_expanded_reproduction.py
```

这些命令可在保留本轮权重的本机重新核验。结果文件绑定 checkpoint 哈希；异机重新训练应使用新的协议与独立结果目录，不能覆盖归档来伪造逐位复现。

原概率文件不改。Power Grid 的作者配置缺少 sample_ratio，显式补为 .2；run.py 将种子 ID 转成 Python int 后保存，避免 NumPy int32 无法 JSON 序列化。这两项不改变表示或选择目标。实验调度器在初始化 PyTorch/OpenMP 前启动独立训练子进程，避免本机嵌套 OpenMP 共享内存错误。

一致性核验通过，共9个新模型，已归档结果对应 21,500 轮非复用模拟（不含失败或中断尝试）。同一评估进程内对相同集合复用结果并记录来源；并行评估进程中可能重复模拟同一 Degree 集合。保留训练命令、checkpoint、独立缓存、图和概率文件哈希、全部种子与逐轮结果。选择耗时只有单次测量，不能作为稳健速度优势的证据。模拟器沿用作者每次最多100传播步。配对指相同初始随机流，不保证逐边耦合。

另外，在全部12个新旧checkpoint上关闭结构项后，重构选择器与原ASW种子集合完全一致，核验了这些模型上的实现差异。记录见 results/HIM/geometric_equivalence_2026-10-30.json。12项 HIM 行为与统计检查通过；这不是全仓库测试声明。详见 results/HIM/expanded_protocol_2026-10-30.json、expanded_summary_2026-10-30.json 与 expanded_verification_2026-10-30.json。原模型机制及公式来源为 methods/HIM/HIM.pdf 和作者源码，ASW-J 的完整定义另见 STRUCTURAL_ASW.md。

## 汇报建议

同一份 HIM.pptx 按约五分钟组织。可讲：问题与模型、ASW、复现与设置、多网络结果、Ciao迁移、ASW-J及负结果。不要逐项读表或所有小数。最后说明完成了哪些验证、哪些推测仍未验证。与组内其他算法的正式排名仍需对齐传播机制和目标，尤其 GBIM 的用户—信息对与普通用户覆盖目标不同。

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
