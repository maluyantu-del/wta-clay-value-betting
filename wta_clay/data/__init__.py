"""Raw data ingestion layer.

Modules:
  loaders      -- low-level file discovery + CSV/Excel reading (no cleaning).
  sackmann     -- Sackmann tennis_wta specific cleaning & clay filtering.
  oddsfeed     -- Tennis-Data.co.uk odds cleaning.
  name_matcher -- fuzzy match between Sackmann player ids and Tennis-Data names.
  merge        -- combine cleaned sources into the master match table.
"""
