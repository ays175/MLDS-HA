#!/usr/bin/env python3
"""
Cross-platform sentiment backfill script.
Populates missing *_sentiment fields using the HybridSentimentAnalyzer.
"""
import asyncio
import sys
from pathlib import Path
from typing import List

import weaviate
import weaviate.classes.query as wvq

# Ensure project root on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cross_platform.sentiment_analyzer import HybridSentimentAnalyzer


def preflight_check(require_transformers: bool = True, require_vader: bool = True) -> None:
    """Ensure RoBERTa transformers and VADER are available before running."""
    tf_ok = False
    vader_ok = False
    # Check transformers
    try:
        from transformers import pipeline  # type: ignore
        _ = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        tf_ok = True
    except Exception:
        try:
            from transformers import pipeline  # type: ignore
            _ = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            tf_ok = True
        except Exception:
            tf_ok = False
    # Check VADER
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        _ = SentimentIntensityAnalyzer()
        vader_ok = True
    except Exception:
        # Try downloading the lexicon automatically
        try:
            import nltk  # type: ignore
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
            _ = SentimentIntensityAnalyzer()
            vader_ok = True
        except Exception:
            vader_ok = False

    missing = []
    if require_transformers and not tf_ok:
        missing.append("RoBERTa transformers")
    if require_vader and not vader_ok:
        missing.append("VADER")
    if missing:
        raise RuntimeError(f"Missing required sentiment components: {', '.join(missing)}. Install dependencies or relax requirements.")

PLATFORMS = [
    {
        "name": "facebook",
        "collection": "FacebookPost",
        "text_fields": ["content_summary", "facebook_content"],
        "sentiment_field": "facebook_sentiment",
    },
    {
        "name": "instagram",
        "collection": "InstagramPost",
        "text_fields": ["content_summary", "instagram_content"],
        "sentiment_field": "instagram_sentiment",
    },
    {
        "name": "tiktok",
        "collection": "TikTokPost",
        "text_fields": ["content_summary", "tiktok_content"],
        "sentiment_field": "tiktok_sentiment",
    },
    {
        "name": "customer_care",
        "collection": "CustomerCareCase",
        "text_fields": ["content_summary", "description", "subject", "comments"],
        "sentiment_field": "sentiment_score",
    },
]


async def backfill_platform(client, analyzer: HybridSentimentAnalyzer, cfg: dict, batch_size: int = 100):
    collection = client.collections.get(cfg["collection"])
    # Fetch objects missing sentiment
    iterator = collection.iterator()
    to_update = []
    for obj in iterator:
        props = obj.properties
        if cfg["sentiment_field"] in props and props[cfg["sentiment_field"]] is not None:
            continue
        text = ""
        for tf in cfg["text_fields"]:
            if props.get(tf):
                text = props.get(tf)
                break
        if not text:
            continue
        to_update.append((obj.uuid, text))
        if len(to_update) >= batch_size:
            texts = [t for _, t in to_update]
            scores = analyzer.analyze_batch(texts, batch_size=64)
            for (uuid, _), score in zip(to_update, scores):
                try:
                    collection.data.update(uuid=uuid, properties={cfg["sentiment_field"]: float(score)})
                except Exception:
                    pass
            to_update = []
    if to_update:
        texts = [t for _, t in to_update]
        scores = analyzer.analyze_batch(texts, batch_size=64)
        for (uuid, _), score in zip(to_update, scores):
            try:
                collection.data.update(uuid=uuid, properties={cfg["sentiment_field"]: float(score)})
            except Exception:
                pass


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sentiment backfill across platforms")
    parser.add_argument("--platforms", nargs="+", choices=["facebook", "instagram", "tiktok", "customer_care", "all"], default=["all"])
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    # Ensure models available
    preflight_check(require_transformers=True, require_vader=True)

    selected = PLATFORMS if "all" in args.platforms else [p for p in PLATFORMS if p["name"] in args.platforms]
    client = weaviate.connect_to_local(skip_init_checks=True)
    try:
        analyzer = HybridSentimentAnalyzer()
        for cfg in selected:
            print(f"🔄 Backfilling sentiment for {cfg['name']}...")
            await backfill_platform(client, analyzer, cfg, batch_size=args.batch_size)
            print(f"✅ Done: {cfg['name']}")
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
