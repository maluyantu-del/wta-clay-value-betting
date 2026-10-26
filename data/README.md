# 数据目录

仓库不含数据。运行 `examples/rebuild_pipeline.py` 前按下列结构放置原始数据:

```
data/raw/sackmann/       克隆 https://github.com/JeffSackmann/tennis_wta
                         需要 wta_matches_YYYY.csv、wta_matches_qual_itf_YYYY.csv、
                         wta_rankings_*.csv、wta_players.csv
data/raw/tennis_data/    从 http://www.tennis-data.co.uk/alldata.php 下载 WTA 2007–2024 年文件
                         (2007–2012 为 .xls,2013 起为 .xlsx)
```

重建后生成:

```
data/interim/name_match_manual_review.csv   未自动匹配、待人工复核的比赛
data/processed/clay_matches_master.parquet  主表
data/processed/features/*.parquet           六个特征文件及其元数据
data/processed/audit/leakage_audit_latest.json
```

Sackmann 数据采用 CC BY-NC-SA 4.0 许可。
