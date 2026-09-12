# 第四问：队友 Q3 改进迁移版

本目录将 `paper_q3_improvements/load_bias_2h` 的“最近 2 小时负荷残差中位数 + 指数衰减”改进迁移到第四问。

- 代码与原第四问隔离：`03_代码/问题4_队友Q3优化/`
- 结果与原第四问隔离：`04_结果/问题4_队友Q3优化/`
- Q4-2：0:00 制定全天计划；该分支不存在日内负荷偏差修正，作为同一框架下的日初方案。
- Q4-3：0:00、6:00、12:00、18:00 更新；后三个时点使用队友的 2 小时因果负荷偏差修正。
- 每次日内更新均继承上一轮有效承诺；上调量和取消量逐次累计结算，不允许用最终净调整抵消中间交易。
- 附件4电价：计划阶段用 7 日前同槽价格预测，真实价格只用于事后结算，避免未来信息泄露。

全年唯一运行命令：

```powershell
D:\Anaconda\python.exe 03_代码\问题4_队友Q3优化\q4_from_q3_improvements.py --part all --start-day 31 --end-day 365 --variant P1
```

快速单日检查：

```powershell
D:\Anaconda\python.exe 03_代码\问题4_队友Q3优化\q4_from_q3_improvements.py --part all --start-day 31 --end-day 32 --variant P1 --output-dir 04_结果\问题4_队友Q3优化_单日检查
```

正式结果中分别生成逐时段 CSV、每日指标、指定日期结果、总体验证、复现清单以及按附件5模板填写的 `result4-2.xlsx`、`result4-3.xlsx`。
