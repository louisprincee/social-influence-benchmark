# 团队共享数据

本目录保存共同数据的来源、处理规则、统计和实际编写的数据转换脚本。原始文件放 `raw/`，处理后的图和节点映射放 `processed/`；需要时再创建，两者默认不提交 Git。各算法作者数据保存在 `methods/方法名/data/`。

当前候选为 [SNAP Email-Eu temporal Department 3](https://snap.stanford.edu/data/email-Eu-core-temporal.html)。已有试验保留 89 名观测用户，取前 180 天、保留至少两次事件的有向关系，得到 464 条边和 15 个孤立点。来源、SHA-256、配置与统计见 [email_eu_dept3_existing_pilot.json](email_eu_dept3_existing_pilot.json)。

这份候选尚未接入三种方法，不是已完成的统一比较数据。HIM 当前按无向图学习；GBIM 需要用户—信息结构，须约定转换规则。

确定数据后记录：获取日期、原始文件哈希、时间窗口、过滤/去重规则、图方向、孤立节点处理、统一节点映射，以及每种方法的输入转换。三人使用同一节点集合和评估口径。
