# 团队共享数据

团队主数据集选用 Ciao 社交推荐数据：信任网络用于 HIM/DeepSN 的用户级任务，评分关系用于 GBIM 的用户—物品扩展。官方来源页面称数据于 2011 年 5 月抓取，并说明当前可下载的是部分发布版本。数据字段、SHA-256、审计统计和使用限制记录在 [ciao_source_manifest.json](ciao_source_manifest.json)；原始 `rating.mat` 和 `trustnetwork.mat` 保存在 `raw/ciao/ciao/`，处理表输出至 `processed/ciao/`。`raw/` 和 `processed/` 均由 Git 忽略。

安装共享预处理脚本依赖后运行：

```bash
python -m pip install -r data/requirements.txt
python data/process_ciao.py
```

脚本不修改原始文件，并会拒绝覆盖已有处理输出。它生成稳定的零起始 user/item ID 映射、有向信任边、供用户级方法使用的无向去重投影，以及保留源顺序、重复行、零分和其他字段的用户评分表。方法专属过滤应在各自 adapter 中进行，并记录在实验配置中。GBIM 原仓库的 Ciao 预处理会按 user-item 对保留首条记录、去除零分，并把信任边对称化；该行为与共享原始导出分开记录。

HIM adapter 可在仓库根目录运行 `methods/HIM/.venv/bin/python methods/HIM/dataload/process_ciao.py`，会验证共享 CSV 的哈希后生成 HIM 图和固定 WIC 概率文件。当前 HIM pilot 使用对称加权级联概率 `p(u,v)=1/max(degree(u), degree(v))`，只作为 Ciao 拓扑上的合成扩散设置，不声称是真实传播概率。训练种子、预算、smoke/full cascade 数量及独立评估随机流约定见 [ciao_wic_protocol.json](ciao_wic_protocol.json)。跨算法正式比较前，团队须共同确认并冻结同一传播协议。

基线 MAT 版本没有评分时间戳，trust 关系也没有建立时间；另有一个较小的评分时间戳文本版本，但未与基线混合。官方页面没有发现明确的数据许可证或再分发许可。原始数据不提交 Git、不对外重新分发；正式公开项目前应向课程教师确认课程使用要求。Ciao 的 trust/rating 记录不是信息传播级联或真实意见采纳标签，因此 HIM、DeepSN 的用户级传播评估必须标为在该静态拓扑上生成的合成扩散实验。

[SNAP Email-Eu-core Department 3](https://snap.stanford.edu/data/email-Eu-core-temporal.html) 仅保留为既有解析/流程 pilot，不再作为主数据集；其统计见 [email_eu_dept3_existing_pilot.json](email_eu_dept3_existing_pilot.json)。
