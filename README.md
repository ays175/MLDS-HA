MLDS-HA

Quick commands

- Install deps (uv):
  uv sync
- Run TikTok ingest:
  uv run python tiktok/tiktok_ingest.py --input data/raw/tiktok/your.csv
- Export TikTok metrics:
  uv run python tiktok/tiktok_metrics_export.py
- Run TikTok agent (all):
  uv run python tiktok/tiktok_agent.py --all
- Build TikTok dashboard bundle:
  uv run python dashboard/tiktok_bundle.py
- Serve dashboard:
  cd dashboard && python3 -m http.server 8000

- Run Facebook ingest:
  uv run python facebook/facebook_ingest.py --input data/raw/facebook/your.csv
- Export Facebook metrics:
  uv run python facebook/facebook_metrics_export.py
- Run Facebook agent (all):
  uv run python facebook/facebook_agent.py --all
- Build Facebook dashboard bundle:
  uv run python dashboard/facebook_bundle.py

Environment

- Copy .env.example to .env and fill values.

Notes

- Large local dirs are kept out of git: WEAVIATE/, Archived_metrics_insights/, Parsers/, data/raw/, data/processed/


