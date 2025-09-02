#!/usr/bin/env python3
"""
Hybrid sentiment analyzer (RoBERTa + VADER ensemble) with safe fallbacks and batching.
Usage: from cross_platform.sentiment_analyzer import HybridSentimentAnalyzer
"""
from typing import List


class HybridSentimentAnalyzer:
    def __init__(self, use_transformers: bool = True, confidence_blend_threshold: float = 0.7, blend_weight_roberta: float = 0.8):
        self._vader = None
        self._tf_pipeline = None
        self._conf_thresh = float(confidence_blend_threshold)
        self._blend_w = float(blend_weight_roberta)
        # Initialize VADER lazily
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
            self._vader = SentimentIntensityAnalyzer()
        except Exception:
            self._vader = None
        # Initialize transformers conditionally
        if use_transformers:
            try:
                from transformers import pipeline  # type: ignore
                # Prefer a robust RoBERTa model
                self._tf_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            except Exception:
                try:
                    from transformers import pipeline  # type: ignore
                    self._tf_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
                except Exception:
                    self._tf_pipeline = None

    def _map_roberta(self, label: str, score: float) -> float:
        l = (label or "").lower()
        s = float(score or 0.0)
        if "positive" in l:
            return min(1.0, s)
        if "negative" in l:
            return max(-1.0, -s)
        return 0.0

    def _vader_score(self, text: str) -> float:
        if not self._vader:
            return 0.0
        try:
            scores = self._vader.polarity_scores(text)
            return float(scores.get("compound", 0.0))
        except Exception:
            return 0.0

    def analyze_text(self, text: str) -> float:
        """Return ensemble sentiment in [-1,1]."""
        if not text or not isinstance(text, str):
            return 0.0
        # Transformers primary
        if self._tf_pipeline is not None:
            try:
                res = self._tf_pipeline(text[:512])[0]
                r_score = self._map_roberta(res.get("label"), float(res.get("score") or 0.0))
                # Blend when low confidence or neutral
                if float(res.get("score") or 0.0) < self._conf_thresh or abs(r_score) < 0.1:
                    v = self._vader_score(text)
                    return max(-1.0, min(1.0, self._blend_w * r_score + (1 - self._blend_w) * v))
                return r_score
            except Exception:
                pass
        # VADER fallback
        return self._vader_score(text)

    def analyze_batch(self, texts: List[str], batch_size: int = 64) -> List[float]:
        if not texts:
            return []
        # If transformers pipeline supports batching, use it; else loop
        if self._tf_pipeline is not None:
            results: List[float] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                try:
                    tf_out = self._tf_pipeline([t[:512] if isinstance(t, str) else "" for t in chunk])
                    # Blend with VADER when low confidence/neutral
                    for j, out in enumerate(tf_out):
                        r_score = self._map_roberta(out.get("label"), float(out.get("score") or 0.0))
                        if float(out.get("score") or 0.0) < self._conf_thresh or abs(r_score) < 0.1:
                            v = self._vader_score(chunk[j] if isinstance(chunk[j], str) else "")
                            results.append(max(-1.0, min(1.0, self._blend_w * r_score + (1 - self._blend_w) * v)))
                        else:
                            results.append(r_score)
                except Exception:
                    # Fallback to per-text
                    results.extend([self.analyze_text(t) for t in chunk])
            return results
        # No transformers: VADER per-text
        return [self._vader_score(t if isinstance(t, str) else "") for t in texts]
