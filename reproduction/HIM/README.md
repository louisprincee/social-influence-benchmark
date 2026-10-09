# 刘同学：HIM 复现入口

作者仓库：[PlaymakerQ/HIM](https://github.com/PlaymakerQ/HIM)。固定提交为 `24e97779f8882666ab70fd0cf19f72440122db9d`。

- 作者代码与运行目录：`methods/HIM`。
- 本机环境入口：`reproduction/HIM/.venv`。
- 修正补丁：`reproduction/HIM/reproduction.patch`。
- 完整步骤：[复现实验指南](复现实验指南.md)。
- 已有实验核验：[实验结果检查](实验结果检查_2026-10-09.md)。
- 已有成果：[结果说明](../../experiments/results/native/HIM/README.md)。

## 你的工作已经迁移

当前这台电脑的 methods/HIM 已包含原来两项代码修正：种子比例按浮点数解析、传播模型不在没有显式种子时重置 NumPy 随机流。原来的 data、configs、save 和 local_checks 已复制并逐文件核对；不是从零重新开始。

最后轮次模型与完整日志位于：

```text
methods/HIM/save/cora_ml/1009_150839_HIM_WIC_10.0/
├── model_last_epoch.pt
├── run.log
├── seeds_last_epoch.json
└── final_evaluation.json
```

已验证的 Cora-ML WIC/ASW 实验：281 个种子，训练种子 42、最终评估种子 1042，最后轮次 54，1000 次独立模拟平均激活 1057.107 个节点，覆盖率 37.619466%。不需要为证明目录迁移而重新训练。

## 本机继续实验

从团队仓库根目录执行：

```bash
source reproduction/HIM/.venv/bin/activate
cd methods/HIM
python --version
```

本机环境通过软链接使用原独立 HIM 环境，原环境目录需保留。新电脑没有这个链接，按详细指南安装 Python 3.12 环境、应用补丁、准备数据和缓存。

数据、权重、缓存不随 GitHub 提交上传。版本、补丁、种子集、指标、审计和英文综述已纳入团队目录；完整文件清单见结果目录中的 artifact_manifest.json。
