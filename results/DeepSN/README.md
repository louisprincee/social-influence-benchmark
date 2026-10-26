# 李同学的 DeepSN 已验证成果

## 原生 Jazz + SIS

| 文件 | 内容 |
|---|---|
| jazz_SIS_k20_native_20261025T183852Z.json | 作者协议：Jazz、SIS、20% 种子、50+200 epoch、100 次评估 |
| jazz_SIS_k20_smoke_20261025T183814Z.json | 1+1 epoch、5 次评估的管道检查，不是正式结果 |
| environment.json | Python 3.11.17、Torch 2.5.1、macOS arm64 CPU |

正式运行：`python model.py -d jazz -dm SIS -sp 20 --tag native`。198 个节点、40 个种子、训练种子 42、CPU、约 25 秒。100 次 SIS 平均覆盖率 **86.4495%**（标准差 2.173 个百分点）。这是 DeepSN 自带 SIS 模拟器上的结果，传播机制与 HIM 的 WIC 不同。

## Ciao 合成 WIC（与 HIM 协议对齐）

共享图哈希与 HIM 协议一致。主结果：30 条训练级联、种子 42/43/44、独立 WIC 1000 轮、评估种子基数 1042。

| 文件 | 内容 |
|---|---|
| ciao_wic_three_training_seeds_2026-10-26.json | 三种子汇总与 t 区间 |
| ciao_wic_independent_n30_s42.json 等 | 各训练种子的 1000 轮 WIC |
| ciao_IC_k5_n30_s42.json 等 | 各训练种子的训练记录与种子集 |
| ciao_IC_k5_ciao-lite_*.json | 16 对、单种子的早期管道结果 |

三种子平均激活 **595.05**（sd 20.30），覆盖率 **8.132%**，95% t 区间 [544.63, 645.47]（df=2）。HIM-ASW 同协议约 735.03 / 10.05%。不要据此排名。
