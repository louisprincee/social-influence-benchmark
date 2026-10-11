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

逐种子结果、完整 replicate 数据与哈希保存在 [`../../results/HIM/ciao_wic_three_training_seeds_2026-10-11.json`](../../results/HIM/ciao_wic_three_training_seeds_2026-10-11.json) 及同目录的逐种子 JSON。下一步是补充 Degree/Random 控制，并与 DeepSN 对齐共同用户级评估协议。

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
