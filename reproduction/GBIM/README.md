# 姜来同学：GBIM 复现入口

作者仓库：[zirui-yuan/GBIM](https://github.com/zirui-yuan/GBIM)。固定提交为 `d5bd89880dfd6d56275be7b485a19b50f461bba3`。

## 你负责的目录

| 内容 | 团队根目录下的路径 |
|---|---|
| 作者仓库 | methods/GBIM |
| 实际运行目录 | methods/GBIM/GBIM |
| 环境、补丁、复现笔记 | reproduction/GBIM |
| 独立环境建议位置 | reproduction/GBIM/.venv |
| 原生结果 | experiments/results/native/GBIM |
| 英文报告 | reports/GBIM |

内层 GBIM 是作者自己的目录结构。当前完成代码下载与版本固定；环境与训练尚未由团队验证。

## 开始顺序

1. 建立独立环境，检查作者 requirements.txt。其 CUDA 扩展和旧 NumPy/SciPy 版本不能直接套到 Apple Silicon；请记录实际验证通过的平台和依赖。
2. 先核对仓库中 LastFM 数据、训练样本和模型输出目录是否齐全，复现默认 Multi-IC、k=5 的任务。
3. 运行目录必须是 methods/GBIM/GBIM；作者路径相对于这里解析。
4. 若选择其他数据，按作者 README 下载和整理，再生成初始训练样本。
5. 保存优化过程、运行时间、随机种子和最终用户—信息种子对，并复制小型结果到原生结果目录。

作者 README 的示例命令如下，**以安装环境、准备 Ciao 数据及输出目录为前提**，当前团队尚未验证它们跑通：

```bash
cd methods/GBIM/GBIM
python datasample.py -d ciao -dm LT -k 5
python main.py -d ciao -dm LT -k 5
```

跑通后，请在这里补充本机可执行的完整命令与依赖版本。

## 与其他两种方法比较时的关键问题

GBIM 选择的是 (用户, 信息) 种子对，在用户网络与信息关联网络中进行 Multi-IC/Multi-LT 传播；目标不是普通 IC 中的唯一激活用户数。因此先完成原问题复现，不直接把它的传播值放进三方法排名。

把信息数量设为 1 也不是直接可用的适配：作者候选生成会抽取 k 个不同信息，k>1 时无法这样采样；同时需检查编号 0 与“未选种子”的编码冲突。统一任务适配要单独验证、记录，不默认为已经完成。
