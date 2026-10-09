# 各方法的复现材料

所有路径相对于团队仓库根目录。

| 方法 | 作者代码 | 中文指南 | 独立环境 |
|---|---|---|---|
| HIM | methods/HIM | [HIM](HIM/README.md) | reproduction/HIM/.venv |
| DeepSN | methods/DeepSN | [DeepSN](DeepSN/README.md) | reproduction/DeepSN/.venv |
| GBIM | methods/GBIM | [GBIM](GBIM/README.md) | reproduction/GBIM/.venv |

这里存放环境版本、复现步骤、代码修改补丁和适配说明。作者代码直接放在 methods 中，团队公共评估则放在 experiments 中。

刘同学本机的 HIM 环境已通过软链接接入，原成果已同步到 methods/HIM。DeepSN 与 GBIM 环境尚未配置。其他电脑不会从 GitHub 获得这个本地软链接或虚拟环境，应按方法独立安装。

**如何保存代码修改：**主仓库提交只保存子模块的提交版本，不能自动保存子模块里的未提交修改。应将修改导出为本目录下的补丁，例如在团队根目录执行：

```bash
git -C methods/HIM diff -- run.py modules/diffusion/DiffusionModel.py > reproduction/HIM/reproduction.patch
```

其他方法同理，先指定实际修改过的代码文件，不要把缓存、数据或权重混入补丁。提交前按[协作指南](../docs/三人协作与提交指南.md)检查。
