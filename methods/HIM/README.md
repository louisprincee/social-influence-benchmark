# HIM：环境、运行与实验记录

本目录包含 HIM 作者代码及课程复现实验适配。负责人：刘晟祺。

## 来源与论文

- 作者仓库：[PlaymakerQ/HIM](https://github.com/PlaymakerQ/HIM)，基准提交 `24e97779f8882666ab70fd0cf19f72440122db9d`。
- 论文：[Influence Strength Estimation in Hyperbolic Space for Social Influence Maximization](https://arxiv.org/abs/2502.13571)，TKDE 2026，DOI [10.1109/TKDE.2026.3683562](https://doi.org/10.1109/TKDE.2026.3683562)。
- 本地修改：为 `run.py` 的 `--seed_ratio` 添加浮点参数类型和 `--data_name`、`--seed_num`、`--seed` 参数；保存最佳 checkpoint 和对应 seed set；令 HIM 训练缓存区分数据图、传播概率、seed budget、样本数和随机种子。`DiffusionModel.py` 仅在显式给定种子时重置 NumPy RNG。
- 公开仓库未为上游代码另行添加许可证；代码来源与许可应以作者仓库为准。

## 环境

团队已在 Apple Silicon macOS、Python 3.12.14、CPU 上安装并运行以下锁定依赖。其他平台需自行验证兼容性。

```bash
python3.12 -m venv methods/HIM/.venv
source methods/HIM/.venv/bin/activate
python -m pip install -r methods/HIM/requirements.txt
python -m pip check
```

本机已配置 `methods/HIM/.venv`；虚拟环境目录由 Git 忽略。不要安装作者原始 `requirements.txt` 覆盖此环境：它包含为旧版 CUDA/PyTorch 准备的依赖。

## 作者数据与配置

HIM 原始数据与配置包不纳入 Git。需要首次运行作者数据集时，从仓库根目录下载基准提交中的归档：

```bash
curl -fL https://raw.githubusercontent.com/PlaymakerQ/HIM/24e97779f8882666ab70fd0cf19f72440122db9d/data.zip \
  -o methods/HIM/data.zip
curl -fL https://raw.githubusercontent.com/PlaymakerQ/HIM/24e97779f8882666ab70fd0cf19f72440122db9d/configs.zip \
  -o methods/HIM/configs.zip
mkdir -p methods/HIM/data methods/HIM/configs
unzip -n methods/HIM/data.zip -d methods/HIM/data
unzip -n methods/HIM/configs.zip -d methods/HIM/configs
```

归档、原始数据、概率库、级联缓存、模型和日志均为本地文件，不要提交。课程共享 Ciao 图由仓库根目录的共享预处理脚本生成；步骤及数据限制见 [`../../data/README.md`](../../data/README.md)。

## 运行

从仓库根目录激活环境后，在 `methods/HIM` 目录执行脚本：

```bash
source methods/HIM/.venv/bin/activate
cd methods/HIM

# Cora-ML 作者数据集：WIC + HIM/ASW，10% 种子预算
python run.py --data_name cora_ml --diffusion_model WIC \
  --seed_ratio 10 --num_instances 30 --epochs 200 --R 100 \
  --seed 42 --method ASW
```

`--method ASW` 为 HIM 的重叠感知选择方法；`--method MD` 为按 LDO 排序的消融变体。`--seed_ratio` 用百分数表示；`--seed_num` 可直接指定精确种子数。运行日志和最佳模型/种子集写入 `save/<dataset>/<run>/`。

## Ciao WIC 实验

Ciao 的 trust 图没有真实传播级联。当前使用固定的对称加权级联概率
`p(u,v)=1/max(degree(u), degree(v))`，仅作为真实静态拓扑上的合成传播协议，不能解释为真实舆情或用户采纳数据。配置、概率库 ID 与校验和见 [`../../data/ciao_wic_protocol.json`](../../data/ciao_wic_protocol.json)。

```bash
# 在仓库根目录准备共享 Ciao 图
methods/HIM/.venv/bin/python methods/HIM/dataload/process_ciao.py
cd methods/HIM

# 固定设置：5% = 366 个用户，30 个训练级联实例
python run.py --data_name ciao --diffusion_model WIC --random_type random \
  --seed_ratio 5 --seed_num 366 --num_instances 30 --epochs 200 --R 100 \
  --seed 42 --method ASW
```

每个训练随机种子应使用独立级联缓存。当前缓存键包含图哈希、概率库 ID、样本比例、预算、实例数和训练种子，避免把 smoke-test 缓存误用于完整实验。

对已保存的 checkpoint，用同一概率库独立评估 ASW 与 MD：

```bash
python evaluate_checkpoint.py \
  --checkpoint save/ciao/<run>/model_best_epoch.pt \
  --seed_metadata save/ciao/<run>/seeds_best_epoch.json \
  --training_instances 30 --evaluation_seed 1042 --rounds 1000 \
  --output ../../results/HIM/paired_selection_ciao_wic_<run>.json
```

评估脚本为两个方法在每个 replicate 中重置到相同的 Python/NumPy 随机种子，并保存逐轮 spread、种子集、checkpoint、图、概率库和训练缓存哈希。它们是相同初始随机流的配对，不保证不同种子集在每一次边尝试上完全耦合。单 checkpoint 的区间反映 Monte Carlo 波动；跨训练种子的差异须另行统计。

## 已完成结果与限制

- Cora-ML：WIC + ASW，281 个种子，独立 1,000 次评估覆盖率为 **37.619466%**。完整记录见 [`../../results/HIM/README.md`](../../results/HIM/README.md)。
- Ciao：三个独立训练种子（42、43、44），每次 30 个训练级联实例；每个 checkpoint 对 ASW/MD 分别进行 1,000 次配对评估。跨三个训练种子的 ASW−MD 平均差为 **+0.371 个激活用户**，95% t 区间为 **[-0.949, 1.691]**（df=2）。样本数较少、区间包含零，不支持 ASW 优于 MD 的结论。
- 旧的两实例/单 epoch Ciao smoke test 仅用于确认流水线可执行，不应与正式训练结果混为一谈。
- Ciao rating/trust 记录不是级联或真实意见采纳标签；当前 spread 全是合成扩散。Ciao 原始数据也不应由本仓库再分发。

逐种子结果、完整 replicate 数据与哈希保存在 [`../../results/HIM/ciao_wic_three_training_seeds_2026-10-11.json`](../../results/HIM/ciao_wic_three_training_seeds_2026-10-11.json) 及同目录的逐种子 JSON。Degree/Random 控制与 ASW-J 扩展已于 2026-10-29 完成，见下文；与 DeepSN 的共同用户级协议仍需进一步完善。

## 引用

```bibtex
@article{qiao_2026_him,
  title   = {Influence Strength Estimation in Hyperbolic Space for Social Influence Maximization},
  author  = {Hongliang Qiao and Shanshan Feng and Min Zhou and Xutao Li and Yunming Ye and Fan Li and Shuo Shang and Yew-Soon Ong},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  doi     = {10.1109/TKDE.2026.3683562}
}
```

## 2026-10-29：ASW-J 选择器扩展已实现

新增独立模块 `methods/HIM/structural_selection.py`，保留原作者选择器不变。它把几何注意力与开放邻域 Jaccard 重叠混合，alpha 在评估前固定为 0.5；同一表示、预算和原惩罚总量下进行对照。新增 9 项行为测试全部通过。

三个已有 Ciao 模型、每组 1000 轮原模拟器评估及 Degree/Random 控制已完成。ASW-J 平均覆盖率 10.0412%，原 ASW 10.0456%，平均激活差 -0.319 个节点，95% t 区间 [-2.074, 1.436]；不支持稳定收益。该结果是已实现的课程方法扩展与探索性验证，不是已证明的性能改进，也不主张文献首创。完整公式、复现范围、运行方法和限制见 [STRUCTURAL_ASW.md](STRUCTURAL_ASW.md)；完整结果见 `../../results/HIM/structural_asw_ciao_summary_2026-10-29.json`。

## 2026-10-30：三个论文网络的补充复现

新增 Cora-ML、Power Grid、Facebook 各三个训练种子（142/143/144），共9次训练，WIC、10%预算、30级联；每个集合在新随机流上评估500次，比较 ASW、MD、ASW-J、Degree、Random。ASW 平均覆盖率分别为38.1440%、18.7217%、31.8099%。ASW-J 相对原 ASW 的三个网络区间均包含零，Ciao亦如此；Facebook 的 Random 观测均值略高于 HIM。不能称为普遍性能提升或全论文完整复现。

完整范围、论文概率库差异、逐网络结果、运行命令与思考见 [EXPANDED_REPRODUCTION.md](EXPANDED_REPRODUCTION.md)。新实验记录位于 results/HIM/expanded_*2026-10-30.json；9个模型及独立训练缓存保留在本机。新增评估调度器会补上作者 Power Grid 配置缺少的 sample_ratio=.2，种子保存修正支持 NumPy 编号。

## 2026-10-30：累计覆盖选择器 HIM-C

继续实现全局累计一跳覆盖选择，固定已有HIM表示及统一先验参数，并加入不使用HIM先验的CO对照。四网络各三checkpoint、每集合新的500次配对模拟已完成。HIM-C相对ASW的未校正区间在Power Grid、Facebook、Ciao为正，Cora-ML跨零；CO消融未支持HIM先验带来稳定额外收益。收益有条件，不能称为通用的新HIM模型。详见 [CUMULATIVE_COVERAGE.md](CUMULATIVE_COVERAGE.md)。

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

旧的未完成恢复文件没有主实验集合审计，修订后会保守拒绝复用；当前12个完整归档的恢复和核验不受影响。
