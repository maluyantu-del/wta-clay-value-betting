"""Feature engineering layer.

Each feature module reads :file:`data/processed/clay_matches_master.parquet`
and emits a parquet at :file:`data/processed/features/<name>.parquet`
keyed on ``match_id``. The pipeline join is in :mod:`wta_clay.pipeline`.

**Layer invariant:** feature modules must NEVER read raw ``winner_*`` /
``loser_*`` columns from the master. They use ``p1_*``, ``p2_*`` and the
``y`` target only. The matcher's id-sort framing fixes label leakage.
"""
