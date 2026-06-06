"""Allow ``python -m frb_search_pipeline`` when the console script is unavailable."""
from frb_search_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
