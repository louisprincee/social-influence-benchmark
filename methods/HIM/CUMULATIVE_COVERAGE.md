# HIM-C：累计覆盖选择的实现、实验与边界

## 为什么继续修改

ASW-J 只在原窗口内重新分配原有惩罚，选出的种子变化较小，四网络实验未显示稳定收益。因此本轮检验另一种假设：把局部的两节点冗余折扣改成整个已选集合的累计覆盖估计，并允许在全图选择候选，能否改善原ASW的选点效果。

## 方法定义

保留已有HIM表示及训练权重，不重新训练、不更改损失。令 h(v) 为 LDO 的反向分位排序，低LDO对应更高h，相同LDO使用平均名次。固定 p=.1、lambda=.5，对所有网络一致：

```text
q(v,u) = 1                           if u=v
q(v,u) = p*((1-lambda)+lambda*h(v))  if (v,u) is an edge and u!=v
q(v,u) = 0                           otherwise
r_S(u) = product_{v in S}(1-q(v,u))
F(S) = sum_u (1-r_S(u))
Delta(v|S) = r_S(v) + q_v * sum_{u in N(v),u!=v} r_S(u)
```

每轮从全部尚未选择的节点中取边际增益最大的节点，更新剩余未覆盖量。实现利用边际增益随覆盖增加而递减的性质进行懒惰重算。CO（Coverage Only）对照采用完全相同的贪心与覆盖代理，只令lambda=0，去掉HIM先验。

CO保留基础概率p=.1。去掉HIM调节会同时去掉节点间差异并改变平均覆盖强度，因此它不是等平均概率的消融。当前只能判断这套HIM先验配置未显示稳定额外收益，不能由此排除更好校准的表示信息有用。

F是独立一跳尝试的覆盖代理，不是模拟器的真实多跳传播目标。代理上的贪心性质不能转述为真实传播的性能保证。p=.1是事先固定的统一先验，不从评估概率文件读取；不同网络的真实模拟参数可能与它不同。Facebook原图有自环，覆盖代理中忽略自环，因q(v,v)=1已包含自覆盖；原图、HIM表示与评估模拟器均不修改。

## 实验协议

评估前冻结 coverage_protocol_2026-10-30.json。使用已有12个checkpoint，四网络各三个训练模型，预算与前轮一致。每个模型对ASW、HIM-C、CO分别运行500次模拟，新的随机种子基数200042；ASW也重新评估，不把前轮100042或1042随机流的均值拿来相减。相同图和协议下相同的CO集合复用结果，不能把它当作三个训练模型。

选择器只读取社交图和固定HIM表示，不读取真实评估边概率、测试结果或测试级联。原模拟器及最多100传播步保持不变。每轮对方法重置相同初始Python/NumPy随机流，不保证逐边随机数耦合。

## 结果

|网络|原 ASW|HIM-C|纯覆盖 CO|
|---|---:|---:|---:|
|Cora-ML|38.1071%|37.9670%|38.0662%|
|Power Grid|18.7235%|20.5250%|20.6025%|
|Facebook|31.8542%|32.2035%|32.3416%|
|Ciao|10.0150%|10.5933%|10.5762%|

|网络|HIM-C−ASW平均激活差|跨3模型95% t区间|HIM-C−CO平均激活差|跨3模型95% t区间|
|---|---:|---|---:|---|
|Cora-ML|-3.937|[-23.664, 15.790]|-2.788|[-7.027, 1.451]|
|Power Grid|+89.013|[80.360, 97.665]|-3.830|[-5.345, -2.315]|
|Facebook|+20.638|[3.182, 38.094]|-8.161|[-9.484, -6.839]|
|Ciao|+42.315|[35.478, 49.152]|+1.250|[-0.462, 2.962]|

相对原ASW，Power Grid、Facebook、Ciao的未校正95%区间为正，Cora-ML未观察到稳定改善。这是特定图、预算、概率库及已有编码器上的探索性结果。原协议没有多重比较校正，也不是新数据集上的独立确认，不能宣称普遍显著优于原方法。

本轮同时改变了候选范围和选点目标，没有分离这两个变化的贡献；CO消融只隔离HIM先验的作用。

消融更限制了归因：HIM-C在Power Grid、Facebook上低于纯覆盖CO，区间均为负；Cora-ML和Ciao上都没有证据支持HIM先验的稳定额外收益。因此可说“累计覆盖机制在部分设置改善了原ASW”，不能说“证明了HIM先验更有效”或“提出了更优的双曲表示模型”。

## 我们自己的扩展体现在哪里

- 将原ASW的局部、两节点几何折扣改为全局候选上的累计覆盖目标，并把HIM的排序作为有界覆盖先验。
- 实现可检验的HIM-C组合策略和去先验CO消融；预先固定参数、保留失败网络和负向消融结果。
- 利用消融区分性能改善和增益来源：目前观察到的收益不能归因于HIM先验本身。

这些是课程项目的方法组合、实现与实验贡献，不是文献首创声明。贪心影响力最大化、子模覆盖及懒惰求值均有成熟研究。下一步若继续研究，应针对HIM先验如何校准、代理如何表达多跳传播另设训练/验证方案；本轮不按这些测试结果再次调参。

## 运行与检查

```sh
# 各网络独立运行，也可在不同进程中并行
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 methods/HIM/.venv/bin/python methods/HIM/evaluate_coverage_selection.py cora_ml
# 数据集参数也可为 power_grid / facebook / ciao
PYTHONPATH=methods/HIM methods/HIM/.venv/bin/python -m unittest discover -s methods/HIM/tests -v
OMP_NUM_THREADS=1 methods/HIM/.venv/bin/python methods/HIM/verify_coverage_selection.py
```

coverage_summary_2026-10-30.json为四网络汇总，12份coverage_<dataset>_s<seed>_2026-10-30.json保存全部种子和逐轮结果，coverage_verification_2026-10-30.json记录核验。结果绑定checkpoint和输入哈希，异机重新训练应使用新的协议与结果目录。原ASW-J和前轮复现结果不覆盖。

新增8项覆盖选择器测试与原12项HIM测试共20项通过。小图穷举边际增益与懒惰贪心一致；核验器重载全部12个checkpoint，重算HIM-C/CO种子、输入哈希、原始结果均值及置信区间均通过。已归档14,000轮非复用新模拟。根目录 unittest discover 未发现测试，不能据此宣称全仓库测试通过。

## 来源与区别

- HIM原论文及实现：本地 methods/HIM/HIM.pdf，https://arxiv.org/abs/2502.13571 。
- Kempe, Kleinberg, Tardos (2003), Maximizing the Spread of Influence through a Social Network：https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf 。本轮不把该工作的传播保证直接套用到一跳代理。
- Leskovec et al. (2007), Cost-effective Outbreak Detection in Networks：https://las.inf.ethz.ch/files/leskovec07cost.pdf 。懒惰求值与子模覆盖是已有方法。
