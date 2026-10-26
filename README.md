# 社交网络影响力最大化：三人复现项目

“社交网络与舆情分析”课程项目。三人分别复现 HIM、DeepSN、GBIM，再在共同数据与评估协议下开展比较。

## 下载与分工

```bash
git clone https://github.com/louisprincee/social-influence-benchmark.git
cd social-influence-benchmark
```

三个方法都是普通目录，代码已随团队仓库保存。修改后直接用团队仓库的 `git add`、`git commit`、`git push` 提交。

| 成员 | 方法与运行入口 | 中文操作指南 | 已完成状态 |
|---|---|---|---|
| 刘同学 | [methods/HIM/run.py](methods/HIM/run.py) | [HIM 环境与实验说明](methods/HIM/README.md) | Mac CPU 环境、训练、保存模型及独立评估已验证 |
| 李同学 | [methods/DeepSN/model.py](methods/DeepSN/model.py) | [DeepSN 复现指南](methods/DeepSN/复现指南.md) | Mac Air CPU 已验证；Jazz+SIS 原生复现完成；Ciao 已按 HIM 协议完成 30 级联、三种子、独立 1000 轮 WIC |
| 姜同学 | [methods/GBIM/main.py](methods/GBIM/main.py) | [GBIM 复现指南](methods/GBIM/复现指南.md) | 源码已准备；团队环境与训练待验证 |

## 文件放在哪里

```text
social-influence-benchmark/
├── README.md                  # 团队入口、分工与协作约定
├── methods/
│   ├── HIM/                   # 源码、运行/复现说明、已验证依赖
│   ├── DeepSN/                # 源码、中文指南、已验证依赖
│   └── GBIM/                  # 源码、中文指南；main.py 直接在这里
├── data/                      # 共享数据说明与已有邮件网络统计
├── results/HIM/               # 已有结果 JSON、种子集与检查记录
├── results/DeepSN/            # Jazz 原生结果与 Ciao 三种子 WIC 评估
├── reports/project_milestone.tex # 团队 project milestone 初稿
├── reports/HIM/               # 英文个人综述及参考文献
└── reports/DeepSN/            # 中文个人综述、PDF 与文献表
```

各自的虚拟环境放 `methods/方法名/.venv`，结果放 `results/方法名/`，报告放 `reports/方法名/`。DeepSN、GBIM 首次产生成果时再创建对应结果、报告目录。共享处理脚本或评估代码实际写出后再添加，没有预建空壳目录或通用配置框架。

GitHub 包含源码、指南、小型结果和报告。数据包、模型、缓存、日志及虚拟环境默认留在本机，其他电脑按各自指南下载数据、建立环境。

## 刘同学已有成果

HIM 的原数据、配置、缓存、日志、模型和检查文件已同步到 `methods/HIM`；73 个文件有 [SHA-256 清单](results/HIM/artifact_manifest.json)。

- 最后轮次模型：`methods/HIM/save/cora_ml/1009_150839_HIM_WIC_10.0/model_last_epoch.pt`（本机保存，需要时单独共享）。
- [已验证结果与解释](results/HIM/README.md)：Cora-ML、WIC + ASW、281 个种子，1000 次独立模拟覆盖率 **37.619466%**。
- Ciao 已完成 HIM 的 3 个独立训练种子（每次 30 个级联实例）和每个 checkpoint 独立 1000 轮 ASW/MD 配对评估；跨训练种子的差异区间包含零，不能得出优越性结论。详见 [三种子汇总](results/HIM/ciao_wic_three_training_seeds_2026-10-11.json)。
- [英文个人综述](reports/HIM/03_individual_review.tex)。

刘同学在当前电脑继续运行，从团队根目录执行：

```bash
source methods/HIM/.venv/bin/activate
cd methods/HIM
```

本机 `methods/HIM/.venv` 是按 [HIM requirements.txt](methods/HIM/requirements.txt) 重建的 Python 3.12.14 环境。不要把环境目录提交到 Git；组员应按依赖文件在自己的机器安装。源码中的复现修正已直接保存，组员无需另行应用补丁。

## 三人如何协作

首次开始自己的实验时建立分支，例如：

```bash
git switch -c li-deepsn
```

每次开始前同步团队主分支：

```bash
git fetch origin
git merge origin/main
```

完成一项可复查的工作后，提交实际修改的文件（将路径、分支名替换成自己的）：

```bash
git add methods/DeepSN/复现指南.md methods/DeepSN/model.py
git commit -m "记录 DeepSN 复现步骤并修正参数入口"
git push -u origin li-deepsn
```

在 GitHub 创建 Pull Request 合并到 main。共享 `data/` 的规则先在组内对齐；不要替他人调整环境。每次实验记录数据版本、完整命令、依赖版本、随机种子、传播参数、种子集合、运行时间及评估结果。权重和完整日志另行共享，小型结果提交到各自结果目录。

## 统一实验与课程要求

先分别完成作者协议下的复现。共同确定新数据集、节点映射、图方向、传播模型、种子预算和独立评估的随机种子后，再做横向比较。原生结果与统一重评估结果分别标明实验设置，可放在同一方法结果目录的不同运行文件夹中。

DeepSN 默认使用 SIS。入口现已读取命令行；Jazz 原生 SIS 已在 Mac Air CPU 上跑通。Ciao 已与 HIM 对齐图、WIC 概率、预算 366、训练种子 42/43/44 和评估种子 1042；DeepSN 三种子平均覆盖率约 8.13%，HIM-ASW 约 10.05%，因优化器不同且区间很宽，不能写成排名。GBIM 优化的是用户—信息种子对，目标与 HIM 的普通用户覆盖率不同；完成任务适配并验证后才可横向排名。详见各自指南。

老师要求至少三种算法在新的有趣数据集上比较；Ciao 已确定为共享新数据集，HIM 还需 degree/random 基线。DeepSN 的 Ciao 三种子独立评估已完成，正式三人排名仍需冻结评估器并补齐 GBIM。Milestone 为 3–5 页；个人英文综述至少 1000 词、至少 10 篇文献，其中近三年至少 5 篇，与任一其他组员重复文献不超过 3 篇。报告只写实际完成的实验，LaTeX 使用双栏。

## 作者代码来源与本地修改

各方法代码基于以下作者仓库版本整理。公开仓库没有为作者代码另行添加许可证；使用与再分发应遵从各作者仓库的许可及要求。

| 方法 | 作者仓库 | 基准提交 |
|---|---|---|
| HIM | [PlaymakerQ/HIM](https://github.com/PlaymakerQ/HIM) | `24e97779f8882666ab70fd0cf19f72440122db9d` |
| DeepSN | [Aselahp/DeepSN](https://github.com/Aselahp/DeepSN) | `bbc7780e448baf2b7f5bdc08668748c10f286ca1` |
| GBIM | [zirui-yuan/GBIM](https://github.com/zirui-yuan/GBIM) | `d5bd89880dfd6d56275be7b485a19b50f461bba3` |

本地适配摘要：HIM 增加可配置数据集、精确种子数和随机种子入口，保存最佳 checkpoint/种子集，并隔离训练缓存；其扩散代码避免未指定种子时重置 NumPy RNG。DeepSN 改为读取命令行、避免 sheaf 参数解析抢占 argv，并把 sheaf `graph_size` 改成随图变化；作者训练/选点流程未改。GBIM 展平作者目录以便直接从 `methods/GBIM` 运行。各方法的详细变更与复现说明保留在其目录中。作者附带的 `.DS_Store`、Python 编译缓存及不需要的 HIM ZIP 已清理；作者的 `resultslog.txt` 仍留在 GBIM 目录中，不作为团队实验结果。
