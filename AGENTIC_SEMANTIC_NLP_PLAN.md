# Sephora Cross-Platform Agentic NLP/Semantic Analytics – One-Page Plan

## Objectives

- Deliver portfolio-level insights (TikTok, Facebook, Instagram) grounded in statistics, BI, and Beauty BI.
- Use semantic retrieval over vectorized post text + numeric filters for rigorous, reproducible analyses.
- Generate agent outputs (comprehensive, content strategy, performance optimization, posting optimization) and executive artifacts.

## Data and Vectorization (Weaviate)

- Object-level vectors: `TikTokPost`, `FacebookPost`, `InstagramPost` (text2vec-contextionary)
- Vectorized TEXT fields contributing to vectors:
  - Facebook: `facebook_content`, `facebook_post_labels_names`, `labels_text`, `content_summary`
  - Instagram: `labels_text`, `content_summary` (and `instagram_content` retrievable; vectorize if desired)
  - TikTok: `labels_text`, `content_summary` (add caption raw text if available)
- Numeric/date/IDs (not vectorized; used as filters): impressions, views, engagements, completion/story completion, likes, saves, clicks, shares, reactions, hour, day_of_week, brand/content_type UUIDs.

## Retrieval Strategy (Hybrid = Semantic + Filters)

1) Filter candidates by numeric/date facets (e.g., brand, content_type, hour/day, views>threshold).
2) Semantic search over vectorized TEXT to refine to most relevant posts.
3) Fetch raw text and metrics for NLP + statistical aggregation.

## NLP/Stats Toolkit

- Sentiment: VADER or transformer-based (per post text); aggregate by brand/content_type/time.
- Keywords/topics: tokenization, n-grams, TF–IDF, keyphrase extraction, topic clustering (e.g., BERTopic).
- Term analytics: top terms by positive/negative sentiment; TF–IDF per brand/content_type; emerging terms (time windows).
- Statistical rigor: quantiles (p50/p75/p95), confidence intervals (where applicable), robust handling of zeros/missing.

## Metrics Layer (Cross-Platform Exporter)

- Source: direct from Weaviate posts across all platforms.
- Outputs to `metrics/cross_platform/` (JSON+CSV):
  - `cross_platform_dataset_overview_<ts>.json/.csv` (totals, averages, date range, benchmarks)
  - `cross_platform_brand_performance_<ts>.json/.csv`
  - `cross_platform_content_type_performance_<ts>.json/.csv`
  - `cross_platform_temporal_analytics_<ts>.json` + hourly/daily CSV + JSON mirrors
  - `cross_platform_top_performers_<ts>.json/.csv`
  - `cross_platform_worst_performers_<ts>.json/.csv`
  - `cross_platform_per_post_sample_<ts>.json`
  - `cross_platform_ai_agent_guide_<ts>.json`
  - `latest_metrics_summary_cross_platform.json`
- Hygiene: coerce numerics, replace ±inf→NaN, guard divide-by-zero, JSON-safe outputs.

## Agent Layer (Cross-Platform)

- Reads cross-platform metrics (not the DB) for consistency and speed.
- Produces in `insights/cross_platform/`:
  - `ai_insights_cross_platform_{focus}_{dataset_id}_{ts}.json/.md` for focuses: comprehensive, content_strategy, performance_optimization, posting_optimization
  - Executive: `cross_platform_executive_summary_{dataset_id}_{ts}.json`, `cross_platform_executive_report_{dataset_id}_{ts}.md`, `cross_platform_executive_dashboard.json`, `latest_executive_summary_cross_platform.json`
- Prompts grounded in metrics; include recommended actions with measured impact and benchmarks.

## RAG Indexing

- Ingest all cross-platform metrics, insights, and executive artifacts into `SocialAnalyticsDoc`.
- Chunk size ~8k chars; tag with `platform=cross_platform`, `doc_type` (metrics|insight|report|dashboard|guide), `dataset_id`.
- Supports semantic retrieval of KPIs, trends, and narratives across files.

## Optional Enhancements (Nice-to-have)

- Vectorize raw content text everywhere (IG `instagram_content`, TikTok caption) for stronger recall.
- Add `brands_text` and `facts_text` (short KPI sentence) per post to enrich semantic signals.
- Topic models per brand/content_type for Beauty BI themes; time-based drift detection on topics/terms.

## Operational Flow

1) Ingest per-platform data (merge mode) → ensures Weaviate collections + platform metrics.
2) Run cross-platform metrics exporter → `metrics/cross_platform/` + `latest_metrics_summary_cross_platform.json`.
3) Run cross-platform agent (focuses + executive) → `insights/cross_platform/`.
4) Index everything into RAG.
5) Build `portal_bundle_latest.json` (bundle builder) → drive `portal_index.html` UI.

## Quality & Validation

- Consistency checks: dataset_id lineage (per-platform → cross), totals vs. sums, non-negativity of metrics, row counts.
- Benchmarks stability: inspect p95 thresholds over time; alert if drift exceeds tolerance.
- Reproducibility: immutable timestamped files; stable “latest” pointers for UIs and agents.
