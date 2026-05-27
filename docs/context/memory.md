# Memory

- decision: composite rows in Panel sheet identified by Country in {DM, EM, Europe, Asia, World, LatAm}; load_panel returns Universe(countries, composites).
- decision: validators raise DataSourceError for hard failures (zero/negative mcap, bad Segment, date gap > MAX_DATE_GAP_DAYS=3) and return list[str] warnings for soft issues (per-series NaNs).
