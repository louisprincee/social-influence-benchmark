# 实验执行与结果导出

公共批量运行脚本尚未实现。各自先按 reproduction/方法名 下的中文指南运行作者入口。

| 方法 | 作者运行目录（相对于团队根目录） | 入口 |
|---|---|---|
| HIM | methods/HIM | run.py |
| DeepSN | methods/DeepSN | model.py |
| GBIM | methods/GBIM/GBIM | main.py |

不能仅进入目录就假定环境与数据已准备好。刘同学本机 HIM 已接入原环境和原实验；其他方法的状态见各自指南。

后续脚本应负责读取配置、设置随机种子、调用作者程序、导出真实种子及结果、避免覆盖已有运行。每次运行使用唯一的运行编号，输出到 experiments/results/native/方法名/运行编号。统一重评估结果另存 benchmark。
