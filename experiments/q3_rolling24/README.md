# 运行说明

工作目录为仓库根目录。未安装任何新依赖，未改原代码、附件或模板。

本机实验解释器：`C:\Users\wu\AppData\Local\Programs\Python\Python312\python.exe`。默认命令 `python` 指向3.14且缺少SciPy，不能直接用于本实验。新入口在缺少openpyxl时把现有自带运行时的site-packages追加到模块搜索路径，只读取该模块，不复制或修改依赖目录。

用上述解释器依次执行：

```text
python experiments/q3_rolling24/run.py --experiment prepare
python experiments/q3_rolling24/test_contracts.py
python experiments/q3_rolling24/run.py --experiment all24 --days 1
python experiments/q3_rolling24/run.py --experiment block24 --days 1
python experiments/q3_rolling24/run.py --experiment all24 --days 7
python experiments/q3_rolling24/run.py --experiment block24 --days 7
python experiments/q3_rolling24/run.py --experiment all24 --days 28
python experiments/q3_rolling24/run.py --experiment block24 --days 28
python experiments/q3_rolling24/run.py --experiment baseline
python experiments/q3_rolling24/run.py --experiment all24
python experiments/q3_rolling24/run.py --experiment block24
python experiments/q3_rolling24/run.py --experiment updates
python experiments/q3_rolling24/run.py --experiment interpolation
python experiments/q3_rolling24/run.py --experiment final
python experiments/q3_rolling24/verify_results.py
python experiments/q3_rolling24/export_workbook_data.py
```

以上 `python` 是解释器占位符。全年命令会检查单日、7天、2月验证文件。forecast缓存只生成一次；重新运行已完成的相同实验读取结果，不重复求解。全年每天保存JSON检查点，恢复时断言首SOC与上一日真实末SOC一致。结果目录名包含结构、更新时间、插值、校正、求解模式、实现版本和天数。

`--policy K0/K06/K0612/Kall` 可与all24或block24组合，使用完全相同的预测缓存单独运行更新消融。只有enabled更新重新优化和使用新预报；关闭更新的区块延用上一次储能计划。BLOCK允许调整的长度始终是下一6小时。

`baseline` 只记录当前仓库缺少第3问可执行代码的事实；不能用第2问结果或报告数字替代第3问baseline。

输出Excel：使用自带Node和artifact-tool，运行 `build_workbook.mjs`，输出 `results/result3_问题3_结果.xlsx`，不会覆盖模板。构建前需存在最终结果和workbook_data.json。数据表费用是程序结算结果，购电总量使用SUM公式；调整表的全天购电费为0时基准费加全部实际调整净费用，不包括紧急购电费，也不是简单的p*g_final。原模板时间表头错位10分钟，仅修正输出副本。

## 求解表达和试运行记录

LP试运行发现情景紧急购电与共同充电同时出现，按任务要求补充物理互斥。先试过逐次增加约束和每时段共享一个二元变量的大M表达；最后采用等价、较快的紧凑表达：

```text
c[t] <= Q*z[t]
d[t] <= Q*(1-z[t])
g[t]-c[t] >= max_s(n[s,t])*z[t]
```

z=0时c=0且最后一式只是g>=0；z=1时d=0、供电覆盖所有场景净负荷及充电，正紧急电价使h=0。每次最多144个共同二元变量，不按场景扩张，未改变费用目标。仅在初始LP确实违反物理互斥时启动这一MILP。HiGHS解必须成功，最优间隙参数1e-8；没有接受超时未证实的解或分位数替代。

历史试运行目录保留，包含lp、physical无版本号、v2。正式全年度对照统一使用v3，不把开发中不同表达和不同天数混在最终比较中。多重最优解会使相同规划目标对应不同储能/合同路径，从而造成真实回放差异；没有通过挑选开发试验中较低的费用来选求解表达。请使用记录的求解器版本复现。
