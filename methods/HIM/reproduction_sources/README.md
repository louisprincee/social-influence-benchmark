# 历史选择器源码

`structural_selection_2026-10-30.py.txt` 是首轮 ASW-J 实验使用的源码原字节快照，仅用于核对归档中的 SHA-256，不作为当前实现导入。

2026-10-31 修正自环图上的度数归一化和开放邻域定义。当前模块位于 `../methods/HIM/structural_selection.py`。旧逐轮结果与训练指纹没有改写；`../verify_structural_revision.py` 从全部12个检查点重选五种方法，确认主实验集合逐个一致后，记录 `results/HIM/structural_revision_verification_2026-10-31.json`。只有当前源码、原始结果文件均与该审计精确匹配时，缓存校验才允许复用历史结果。

本机的模型、训练缓存和原数据仍不提交。另一台机器需要对应本地归档，或使用独立的新协议重新训练。
