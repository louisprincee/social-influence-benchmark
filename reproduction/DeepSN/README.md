# 李同学：DeepSN 复现入口

作者仓库：[Aselahp/DeepSN](https://github.com/Aselahp/DeepSN)。固定提交为 `bbc7780e448baf2b7f5bdc08668748c10f286ca1`。

## 你负责的目录

| 内容 | 团队根目录下的路径 |
|---|---|
| 作者代码 | methods/DeepSN |
| 原始入口 | methods/DeepSN/model.py |
| 环境、补丁、复现笔记 | reproduction/DeepSN |
| 独立环境建议位置 | reproduction/DeepSN/.venv |
| 原生结果 | experiments/results/native/DeepSN |
| 英文报告 | reports/DeepSN |

目前完成代码下载与版本固定；环境、数据处理和训练尚未由团队验证。表格中的结果和报告目录在首次产生成果时创建。

## 开始顺序

1. 在 reproduction/DeepSN/.venv 建立独立 Python 环境，梳理实际导入的依赖，再记录验证通过的版本。这里尚未给出已验证的安装命令，不能照搬 HIM 的环境锁定文件。
2. 阅读作者入口与数据生成代码，先用作者提供的较小数据完成运行。
3. 处理入口 `parse_args(args=[])`：它会忽略终端传入的参数，README 所说的 `python model.py -h` 也不能可靠显示帮助。先改为读取实际命令行，再验证参数确实生效，把修改存为补丁。
4. 显式选择传播模型。原入口默认 Jazz 数据、SIS、20% 种子比例；不能默认为正在做普通 IC 对比。
5. 检查训练与评估数据划分、节点映射、设备与随机种子，保留命令和日志。
6. 导出最终真实种子集合，再交给团队独立评估。原生结果存入表格所列目录。

进入代码的命令（从团队根目录）：

```bash
cd methods/DeepSN
```

以上只是进入目录，尚不是已验证的训练命令。跑通后请在本指南补充你的完整执行命令、版本及文件格式。

## 与 HIM 对比前必须对齐

作者 IC 模拟使用 `1 / in_degree(target)` 的传播概率；LT 阈值在 [0.3, 0.6] 间采样。它们与现有 HIM 实验协议不同。同名模型也需要核对实际参数、图方向和种子预算，原生指标先独立记录。
