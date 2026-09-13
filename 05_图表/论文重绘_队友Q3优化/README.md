# 论文全量重绘图（Nature Figure Python 后端）

本目录为独立重绘输出，不覆盖原有 `05_图表` 图件。脚本 `redraw_all_figures.py` 从问题 1–4 的结果 CSV 及 `paper_q3_improvements` 误差评估结果读取数据，统一导出 SVG、PDF 和 600 dpi PNG。

## 论文原有图位

`raw_q1_load_pv`、`process_q1_dispatch`、`process_q1_storage`、`result_q1_cost`、`raw_q2_forecast_error_distribution`、`process_q2_solver_modes`、`process_q2_storage_state`、`result_q2_monthly_cost`、`result_q2_specified_dates`、`process_q3_adjustment_epochs`、`result_q3_cost_components`、`result_q3_cumulative_cost_difference`、`process_q3_storage_specified`、`result_q4_total_cost_comparison`。

## 隔离替代/优化方案图

`result_q4_original_vs_optimized` 对比原第四问与基于 `paper_q3_improvements` 的隔离改进方案；`result_q4_optimized_specified` 展示改进方案指定日期的计划、最终承诺和 SOC；`process_q4_load_bias` 展示 2 小时负荷偏差修正前后的 MAE。

每个 PDF 旁附有碰撞审计 JSON 与审计叠加 PDF；多面板图附有 `*.alignment.json` 和对齐叠加 SVG。
