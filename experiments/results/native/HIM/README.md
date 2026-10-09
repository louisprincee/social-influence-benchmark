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

早期及其他四次运行也保留在 methods/HIM/save。代码修正已应用，环境通过 reproduction/HIM/.venv 的本地软链接使用原环境。模型、数据、缓存和虚拟环境默认不随 Git 提交上传；组员需要权重时另行同步，不能以 GitHub 上只有摘要误认为模型已经丢失。

## 已验证的结果如何写入报告

Cora-ML，WIC + ASW，281 个种子，训练种子 42，独立评估种子 1042，最后轮次 54，1000 次模拟平均激活 1057.107 个节点，覆盖率 **37.619466%**。

第43轮最高验证值 38.15% 与第54轮种子集独立评估不是同一指标。当前证据支持这一单次训练的复现结果，不支持 HIM 已显著优于其他方法或已完成论文全部表格。

原审计中的 checked_run、缓存清单中的 file 都相对于作者运行目录。artifact_manifest 的 path 则相对于团队根目录。
