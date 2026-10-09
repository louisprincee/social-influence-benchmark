# 社交网络影响力最大化：三人复现项目

本项目用于“社交网络与舆情分析”课程，分别复现 HIM、DeepSN 和 GBIM，再完成共享数据上的实验分析。

## 三人分工与入口

| 成员 | 算法 | 作者代码 | 环境与指导 | 实验结果 | 报告 |
|---|---|---|---|---|---|
| 刘同学 | HIM | [methods/HIM](methods/HIM) | [HIM 复现指南](reproduction/HIM/README.md) | [HIM 成果](experiments/results/native/HIM/README.md) | [reports/HIM](reports/HIM) |
| 李同学 | DeepSN | [methods/DeepSN](methods/DeepSN) | [DeepSN 复现指南](reproduction/DeepSN/README.md) | 运行后存入 experiments/results/native/DeepSN | 运行后存入 reports/DeepSN |
| 姜来同学 | GBIM | [methods/GBIM](methods/GBIM) | [GBIM 复现指南](reproduction/GBIM/README.md) | 运行后存入 experiments/results/native/GBIM | 运行后存入 reports/GBIM |

所有方法目录采用原方法的名称：**HIM、DeepSN、GBIM**。本文件与团队指导文档中的路径都相对于团队仓库根目录。

## 清晰的目录分工

```text
social-influence-benchmark/
├── methods/                       # 直接进入作者代码
│   ├── HIM/                       # 入口 run.py；本机包含已迁移的实验
│   ├── DeepSN/                    # 入口 model.py
│   └── GBIM/                      # 作者入口 GBIM/main.py
├── reproduction/                  # 三人分别维护环境、补丁和复现指南
│   ├── HIM/
│   ├── DeepSN/
│   └── GBIM/
├── data/                          # 团队共享数据
│   ├── manifests/                 # 来源、哈希、处理规则与统计
│   ├── tools/                     # 数据处理工具的职责说明，代码待实现
│   ├── raw/                       # 本地原始数据，Git 忽略
│   └── processed/                 # 本地处理数据，Git 忽略
├── experiments/                   # 团队共用的实验协议与工具
│   ├── configs/                   # 配置说明及 HIM 已完成实验的参数记录
│   ├── evaluation/                # 评估协议，代码待实现
│   ├── scripts/                   # 脚本入口说明，公共脚本待实现
│   └── results/
│       ├── native/HIM/            # 已验证的 HIM 结果、审计及文件清单
│       └── benchmark/             # 统一比较的准入说明，暂无对比结果
├── reports/HIM/                   # 英文个人综述与参考文献
└── docs/                          # 中文结构说明与三人协作指南
```

目录不再使用空的 `.gitkeep`。Git 不保存空目录，原先这些文件只用于保留规划的目录；现在每个规划目录都通过中文 README 解释其职责与完成状态，不会误认为已有工具。

## 刘同学已有 HIM 成果

此前工作已完整保留。本机的 `methods/HIM` 已应用两项复现修正，并同步了原目录的 `data`、`configs`、`save` 和 `local_checks`；73 个文件已逐一核对 SHA-256。原独立 HIM 目录仍是备份。

- 已验证的最后轮次模型、种子集、完整日志：`methods/HIM/save/cora_ml/1009_150839_HIM_WIC_10.0/`。
- 团队可提交的结果与审计：[experiments/results/native/HIM](experiments/results/native/HIM/README.md)。
- 代码修正：[reproduction/HIM/reproduction.patch](reproduction/HIM/reproduction.patch)。
- 环境入口：`reproduction/HIM/.venv`。在刘同学本机，它通过本地软链接使用原环境，没有移动或重装虚拟环境。
- 已验证结果：WIC + ASW，281 个种子，最后轮次为 54，独立 1000 次模拟覆盖率 **37.619466%**。这是可以写入报告的单次训练结果，不是三方法排名。

**本机保存与 GitHub 上传不同：**数据、训练缓存、模型权重和虚拟环境保存在本机，默认不进入 Git 提交。GitHub 会保存结果 JSON、清单、补丁、环境版本和报告。其他成员下载后需按照指南准备环境和数据；如需这个已训练模型，应另行同步该运行目录中的权重文件。

## 下载与首次开始

```bash
git clone --recurse-submodules https://github.com/louisprincee/social-influence-benchmark.git
cd social-influence-benchmark
git submodule update --init --recursive
```

这三个方法是固定版本的 Git 子模块。点击方法目录就是作者仓库，不再包含团队添加的包装层。GBIM 自带的内层 `GBIM` 目录属于作者原结构。

然后根据自己的方法打开上面的中文复现指南。三个方法使用独立环境，不共用一个 requirements 文件。

刘同学在**当前这台电脑**可以直接从团队根目录进入已有环境和代码：

```bash
source reproduction/HIM/.venv/bin/activate
cd methods/HIM
python --version
```

这一步不会重新训练，也不会清空已有结果。具体实验命令见 [HIM 详细指南](reproduction/HIM/复现实验指南.md)。

## 实验如何独立开展，再汇总

1. 各自先完成作者协议下的复现，代码修改保存成 `reproduction/方法名/` 下的补丁。
2. 各自的原生结果存到 `experiments/results/native/方法名/`；每次运行保留参数、随机种子、日志和种子集。
3. 共同确定新数据集、节点映射、传播模型、种子预算和评估协议。
4. 对齐任务后，统一重评估的结果才进入 `experiments/results/benchmark/`。

DeepSN 与 GBIM 的团队环境和训练尚未验证。DeepSN 默认 SIS 且入口忽略命令行参数；GBIM 优化用户—信息种子对，不能直接拿其 Multi-IC/Multi-LT 数值与 HIM 的普通用户种子覆盖率排名。详细问题见各自指南。

[三人协作与提交指南](docs/三人协作与提交指南.md)说明分支、子模块补丁、文件存放和课程材料要求。
