#!/usr/bin/env Rscript
# Independent R/ggplot2 test track for the strict 10-scenario run.
# Rendering requires Rscript, ggplot2, patchwork, svglite and ragg.
root <- "D:/Git/mm"
res <- file.path(root, "04_结果", "问题3_10场景完整求解")
out <- file.path(root, "05_图表", "问题3_10场景测试", "r")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
required <- c("ggplot2", "patchwork", "svglite", "ragg")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
library(ggplot2); library(patchwork)
daily <- read.csv(file.path(res, "问题3_每日指标.csv"), fileEncoding = "UTF-8-BOM")
pair <- read.csv(file.path(res, "问题3_每日配对差.csv"), fileEncoding = "UTF-8-BOM")
daily$date <- as.Date(daily$date); pair$date <- as.Date(pair$date)
theme_set(theme_classic(base_size = 7) + theme(legend.position = "top", panel.grid = element_blank()))
p1 <- ggplot(daily, aes(date, total_cost_yuan/1000, colour=policy)) + geom_line(linewidth=.35) + labs(x=NULL, y="Daily cost (10^3 yuan)", title="Rolling daily operating cost")
p2 <- ggplot(pair, aes(date, H24_minus_H48_yuan)) + geom_hline(yintercept=0, linewidth=.3) + geom_line(colour="#D9822B", linewidth=.35) + labs(x=NULL, y="H24 − H48 (yuan)", title="Paired daily cost difference")
p3 <- ggplot(pair, aes(date, cumsum(H24_minus_H48_yuan)/1000)) + geom_hline(yintercept=0, linewidth=.3) + geom_line(colour="#B24C63", linewidth=.45) + labs(x="Date", y="Cumulative saving (10^3 yuan)", title="Accumulated advantage of H48")
p4 <- ggplot(daily, aes(date, end_storage_kwh/1000, colour=policy)) + geom_line(linewidth=.35) + labs(x="Date", y="End storage (10^3 kWh)", title="Daily terminal storage")
fig <- (p1 | p2) / (p3 | p4) + plot_annotation(tag_levels="a")
ggsave(file.path(out, "r_figure1_yearly_comparison.pdf"), fig, width=183, height=132, units="mm", device=cairo_pdf)
ggsave(file.path(out, "r_figure1_yearly_comparison.svg"), fig, width=183, height=132, units="mm", device=svglite::svglite)
ggsave(file.path(out, "r_figure1_yearly_comparison.tiff"), fig, width=183, height=132, units="mm", dpi=600, device=ragg::agg_tiff)
