# 作者代码来源与修改记录

三个目录保存作者代码的普通文件副本，按以下原始提交确定来源。保留作者 README、原始依赖文件与论文引用信息。公开仓库的代码来源与使用条款应以作者仓库为准，本项目未为作者代码另行添加许可证。

| 方法 | 作者仓库 | 原始提交 |
|---|---|---|
| HIM | https://github.com/PlaymakerQ/HIM | `24e97779f8882666ab70fd0cf19f72440122db9d` |
| DeepSN | https://github.com/Aselahp/DeepSN | `bbc7780e448baf2b7f5bdc08668748c10f286ca1` |
| GBIM | https://github.com/zirui-yuan/GBIM | `d5bd89880dfd6d56275be7b485a19b50f461bba3` |

## 当前修改

- HIM：`run.py` 的 `--seed_ratio` 增加 `type=float`；`modules/diffusion/DiffusionModel.py` 仅在显式提供随机种子时调用 `np.random.seed(seed)`。这两项此前已用于刘同学的实验，现直接保存进团队仓库。
- DeepSN：作者源码未修改。
- GBIM：作者内层 `GBIM/` 的内容移至 `methods/GBIM/`，Python 源码未修改。从 `methods/GBIM` 运行即可保持原有相对数据路径。
- 各方法添加中文复现指南，HIM 另外保存已经验证的 Mac 依赖版本。
- 作者附带的 `.DS_Store` 和 Python 编译缓存已清理；二进制数据、压缩包和权重不提交，获取方式见各方法指南。GBIM 的 `resultslog.txt` 为作者附带记录，不是本团队的复现实验结果。

后续成员直接修改对应目录，用团队 Git 记录修改及原因。更新作者版本时，在这里同步原始提交信息和必要的兼容性说明。
