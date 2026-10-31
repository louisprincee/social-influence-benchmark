# 刘同学的 HIM 已验证成果

你此前的工作已保留并迁入团队目录，没有丢失。

## 可提交到 GitHub 的成果

| 文件 | 内容 |
|---|---|
| final_evaluation.json | WIC/ASW 最后轮次种子集的独立评估结果 |
| seeds_last_epoch.json | 281 个真实节点编号 |
| user_run_audit_2026-10-09.json | 保存模型、重新选点与重新评估的一致性核验 |
| user_run_log_summary_2026-10-09.json | 五次已有运行的日志摘要 |
| cascade_manifest.json | WIC/WLT 训练缓存的原始哈希、样本数和边数 |
| environment.json | 原实验的 Python、依赖、设备及数据统计 |
| artifact_manifest.json | 本次同步的 73 个本地实验文件及 SHA-256 |
| 迁移后验证.json | 从新位置加载模型、重新选择种子并执行1000次独立模拟的结果 |
| paired_selection_ciao_wic_smoke_2026-10-11.json | Ciao 单 checkpoint、ASW/MD 各 1,000 次配对评估；仅为 smoke 结果 |
| paired_selection_ciao_wic_train30_seed42_2026-10-11.json | Ciao 30-instance 训练、单训练种子 checkpoint 的独立 ASW/MD 评估 |
| ciao_wic_three_training_seeds_2026-10-11.json | 三个 HIM 训练种子的 ASW/MD 配对评估汇总；另附逐种子完整结果 |

上述已有结果、种子与审计文件保留原内容；目录整理没有重新生成训练结果。

迁移后已直接从新目录严格加载保存模型，重新选出的281个种子与原集合完全一致；使用1042作为随机种子重新执行1000次模拟，均值1057.107和标准差32.12138775022025均与原结果精确一致。此检查没有重新训练。

## 本机完整文件在哪里

团队根目录下的 methods/HIM 已同步原 data、configs、save 和 local_checks。最后轮次模型位于：

```text
methods/HIM/save/cora_ml/1009_150839_HIM_WIC_10.0/
├── model_last_epoch.pt
├── run.log
├── seeds_last_epoch.json
└── final_evaluation.json
```

早期及其他四次运行也保留在 methods/HIM/save。代码修正已应用，环境通过 methods/HIM/.venv 的本地软链接使用原环境。模型、数据、缓存和虚拟环境默认不随 Git 提交上传；组员需要权重时另行同步，不能以 GitHub 上只有摘要误认为模型已经丢失。

## 已验证的结果如何写入报告

Cora-ML，WIC + ASW，281 个种子，训练种子 42，独立评估种子 1042，最后轮次 54，1000 次模拟平均激活 1057.107 个节点，覆盖率 **37.619466%**。

第43轮最高验证值 38.15% 与第54轮种子集独立评估不是同一指标。当前证据支持这一单次训练的复现结果，不支持 HIM 已显著优于其他方法或已完成论文全部表格。

原审计中的 checked_run、缓存清单中的 file 都相对于作者运行目录。artifact_manifest 的 path 则相对于团队根目录。

## Ciao HIM smoke test（不可作为性能结论）

HIM 的 Ciao adapter 成功读取 7,317 个用户、85,205 条无向信任边，并使用固定概率 ID `ciao_symmetric_weighted_cascade_v1_f2fc09305d5f`。Smoke training 只有 2 个传播实例、1 个 epoch、366 个种子；随后从保存的同一 checkpoint 分别用 ASW 和 MD 选点，在评估种子基数 1042、1,000 个 replicate 下进行配对模拟。

| 选择器 | 平均激活用户 | 覆盖率 |
|---|---:|---:|
| ASW | 714.156 | 9.7602% |
| MD | 713.378 | 9.7496% |

配对差值为 +0.778 个激活用户，95% 配对 $t$ 区间为 $[-0.289, 1.845]$；ASW 较高 517 次、相同 17 次、MD 较高 466 次。区间跨过零，且 checkpoint 只训练了 1 个 epoch，因此这些数字只验证 checkpoint 加载、选点和独立评估链路；它们不能说明 HIM 或 ASW 在 Ciao 上更好，也不能代表真实信息传播。Ciao 的概率规则是静态图上的合成假设。完整逐轮结果、种子 ID 和输入哈希见 JSON artifact。

正式 30-instance 训练前又发现并修复 HIM 训练缓存原本不区分实例数、随机种子和概率库的问题；新缓存键还纳入样本比例和图文件 SHA-256，并在读取时校验级联数量。历史两实例 smoke 文件仍保留在本机 ignored 目录，不会被新设置静默复用。为完成训练，本机 `methods/HIM/.venv` 已替换为按锁文件重建的 Python 3.12.14 环境（目录由 Git 忽略）。

### 首批三个 30-instance runs

在固定 Ciao WIC 设置下完成了训练种子 42、43、44，各使用 30 个传播实例，最多 200 epoch，每 epoch 100 轮在线验证，patience 10。Best checkpoint 分别为 epoch 28、19、26；早停分别在 epoch 39、30、37。ASW 和 MD 每次选出的 366 个种子集在独立评估随机流（基数 1042）上分别配对运行 1,000 次：

| 选择器 | 三个训练种子平均激活用户 | 三个训练种子平均覆盖率 |
|---|---:|---:|
| ASW | 735.034 | 10.0456% |
| MD | 734.662 | 10.0405% |

跨三个训练种子的配对差（ASW--MD）均值为 +0.371 个激活用户，样本标准差 0.531，95% $t$ 区间为 $[-0.949,1.691]$（df=2），包含零。ASW 在两个训练种子的条件均值更高，在一个种子更低。各训练种子的 1,000 次重复给出 Monte Carlo 变化估计；只有三个独立训练种子时，训练间区间仍很不精确。这些结果不支持 ASW 优于 MD，也不构成跨算法排名。模拟基于固定的拓扑派生合成概率，不表示 Ciao 的真实传播。完整训练命令、checkpoint/cache 哈希、各方法种子集及 replicate 数据保存在三份逐 seed JSON 与聚合结果 JSON。

## 整理说明

所有已有结果与审计 JSON 保留原内容。历史清单与迁移检查中的 `reproduction/HIM/.venv` 是当时的环境入口，当前已简化为 `methods/HIM/.venv`；模型、数据与种子集路径保持有效。详细的第3、4步核验见 [实验结果检查](实验结果检查_2026-10-09.md)。

## 2026-10-29：选择器结构扩展与简单基线

已实现课程扩展 ASW-J，固定 alpha=0.5，复用三个已有模型而不重新训练；原 ASW/MD 的种子集合重新计算并与归档核对后，复用相同协议的历史评估。新增 7,000 轮实际传播模拟，用于 ASW-J、Degree 和三个 Random 集合；相同 Degree 集合的结果只模拟一次，跨模型复用并记录来源。

| 方法 | 三模型/种子集合的平均覆盖率 |
|---|---:|
| ASW | 10.0456% |
| MD | 10.0405% |
| ASW-J | 10.0412% |
| Degree | 9.7280% |
| Random | 8.1075% |

ASW-J 在三个模型中各替换 1、2、1 个种子，相对原 ASW 的平均激活差为 -0.319 个节点，95% t 区间 [-2.074,1.436]。不支持稳定增益。Degree 是一个确定性集合，Random 是三个抽样集合，不能把它们当作三个训练模型。新扩展不主张文献首创，原方法的表示学习结构不变。

定义与边界见 `methods/HIM/STRUCTURAL_ASW.md`（从仓库根目录定位）。完整逐轮数据、选择耗时和哈希见 `structural_asw_ciao_seed*_2026-10-29.json`，汇总见 `structural_asw_ciao_summary_2026-10-29.json`，预先固定的协议见 `structural_asw_protocol_2026-10-29.json`，一致性检查见 `structural_asw_verification_2026-10-29.json`。

## 2026-10-30：多网络补充复现

已完成 Cora-ML、Power Grid、Facebook 各三个新训练种子，WIC/10%预算，每次30级联、每集合500次新评估随机流。原ASW平均覆盖率分别为38.1440%、18.7217%、31.8099%。五种选择方法的逐模型原始结果见 expanded_<dataset>_s<seed>_2026-10-30.json，汇总见 expanded_summary_2026-10-30.json，一致性核验见 expanded_verification_2026-10-30.json。

ASW-J 在上述三个网络及 Ciao 的跨训练区间全部包含零。完整解释及与论文设置的差异见 ../../methods/HIM/EXPANDED_REPRODUCTION.md；原2026-10-09单模型结果保留为历史记录，未被新三模型均值替代或覆盖。

## 2026-10-30：HIM-C 新随机流验证

12个已有checkpoint，ASW/HIM-C/CO三方法，各500轮模拟，随机种子基数200042。汇总 coverage_summary_2026-10-30.json，逐模型 coverage_<dataset>_s<seed>_2026-10-30.json，协议 coverage_protocol_2026-10-30.json，核验 coverage_verification_2026-10-30.json。不与之前不同评估随机流的结果直接相减。

HIM-C相对ASW平均激活差：Cora-ML −3.937、Power Grid +89.013、Facebook +20.638、Ciao +42.315；后三个未校正95%跨训练区间为正。HIM-C未显示对CO的稳定额外收益，Power Grid/Facebook上CO反而更高。完整定义与归因限制见 ../../methods/HIM/CUMULATIVE_COVERAGE.md。

## 2026-10-31：选择器修订核验

自环归一化与开放邻域定义修正后，全部12模型的五种主方法选点与历史归档一致。记录见 `structural_revision_verification_2026-10-31.json`。旧结果及其源码哈希保留原值，修正版通过精确文件指纹与集合重选审计绑定，未新增传播模拟，原覆盖率与区间不变。
