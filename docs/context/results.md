# Results

- Task 4 (IO load_panel): roro/io.py exposes load_panel which splits Panel sheet into countries vs composites (DM/EM/Europe/Asia/World/LatAm) and validates required columns; tests use tiny_xlsx fixture in tests/conftest.py.
- Task 6 (Validators): roro/validators.py exposes DataSourceError, validate_universe, validate_prices; hard fails on zero/negative mcap, bad Segment, or contiguous business-day gaps > 3; warns on per-series NaN counts.
