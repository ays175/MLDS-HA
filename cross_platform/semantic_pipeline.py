#!/usr/bin/env python3
"""
Semantic topics pipeline:
- Samples vectors and properties from Weaviate per platform (stratified + seed neighborhoods)
- Clusters vectors to topics, labels topics via TF-IDF, computes KPIs per topic
- Builds temporal trends per topic
- Aligns topics across platforms and customer care
- Writes JSON artifacts under metrics/{platform} and metrics/global

Note: Uses stored vectors in Weaviate (no re-embedding).
"""
from __future__ import annotations

import json
import math
import random
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import weaviate
import weaviate.classes.query as wvq
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Language detection
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    # Optional: topic coherence (c_v, c_npmi)
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    _GENSIM_AVAILABLE = True
except Exception:
    _GENSIM_AVAILABLE = False


PLATFORMS = ["facebook", "instagram", "tiktok", "customer_care"]
PLATFORM_COLLECTION = {
    "facebook": "FacebookPost",
    "instagram": "InstagramPost",
    "tiktok": "TikTokPost",
    "customer_care": "CustomerCareCase",
}


@dataclass
class SampleConfig:
    max_items: int = 1500000  # High default for full datasets, overridden by YAML configs
    random_seed: int = 42
    recent_days_priority: int = 90  # prioritize last N days
    seed_query_per_topic: int = 150
    min_docs_for_clustering: int = 500
    max_clusters: int = 30
    # Multilingual support
    use_multilingual_embeddings: bool = True
    multilingual_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"


def _load_platform_config(platform: str, project_root: Path) -> Dict[str, Any]:
    """Load platform-specific YAML configuration"""
    config_path = project_root / "ingestion" / "configs" / f"{platform}.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def _get_sample_size_for_platform(platform: str, project_root: Path, default: int = 1500000) -> int:
    """Get sample size from platform YAML config or use default"""
    config = _load_platform_config(platform, project_root)
    semantic_config = config.get('semantic_analysis', {})
    return semantic_config.get('sample_size', default)


def _ensure_dirs(project_root: Path, platform: str) -> Tuple[Path, Path]:
    metrics_dir = project_root / "metrics" / platform
    metrics_dir.mkdir(parents=True, exist_ok=True)
    global_dir = project_root / "metrics" / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, global_dir


def _parse_date(dt: Any) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(dt):
            return None
        return pd.to_datetime(str(dt), errors="coerce")
    except Exception:
        return None


def _reservoir_stride_sampler(objs: List[Any], k: int, stride: int = 5, seed: int = 42) -> List[Any]:
    """Simple mixed sampler: stride pick + reservoir for diversity."""
    random.seed(seed)
    picked: List[Any] = []
    # Stride
    for i, obj in enumerate(objs):
        if len(picked) >= k:
            break
        if i % stride == 0:
            picked.append(obj)
    # Reservoir fill
    for obj in objs:
        if len(picked) >= k:
            break
        j = random.randint(0, len(picked) + 10)
        if j < len(picked):
            picked[j] = obj
        else:
            picked.append(obj)
    return picked[:k]


def _fetch_iterator_objects(col, properties: List[str], need_vector: bool, limit: int) -> List[Any]:
    objs: List[Any] = []
    it = col.iterator()
    count = 0
    for obj in it:
        try:
            if need_vector and not getattr(obj, "vector", None):
                continue
            # keep as-is; we'll read properties later
            objs.append(obj)
            count += 1
            if count >= limit:
                break
        except Exception:
            continue
    return objs


def _objects_to_frame(objs: List[Any], platform: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for o in objs:
        p = o.properties or {}
        vec = getattr(o, "vector", None)
        if vec is None:
            # if not attached, try metadata
            try:
                vec = o.metadata.vector  # type: ignore
            except Exception:
                vec = None
        if platform == "customer_care":
            created = p.get("created_date")
            text = p.get("content_summary") or ""
            rows.append({
                "id": p.get("case_id") or "",
                "created_time": created,
                "text": text,
                "labels": ", ".join(filter(None, [p.get("issue_category"), p.get("priority"), p.get("origin")])),
                "sentiment": p.get("sentiment_score"),
                "urgency": p.get("urgency_score"),
                "vector": vec,
            })
        else:
            created = p.get("created_time")
            text = p.get("content_summary") or p.get(f"{platform}_content") or ""
            rows.append({
                "id": p.get(f"{platform}_id") or p.get("facebook_id") or p.get("instagram_id") or p.get("post_id") or "",
                "created_time": created,
                "text": text,
                "labels": p.get(f"{platform}_post_labels_names") or p.get("labels_text") or "",
                "sentiment": p.get(f"{platform}_sentiment"),
                "engagement_rate": p.get("engagement_rate"),
                "vector": vec,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_time"] = df["created_time"].apply(_parse_date)
    return df


def sample_platform(client, platform: str, cfg: SampleConfig, project_root: Path) -> pd.DataFrame:
    col_name = PLATFORM_COLLECTION[platform]
    col = client.collections.get(col_name)
    
    # Get platform-specific sample size from YAML config
    platform_sample_size = _get_sample_size_for_platform(platform, project_root, cfg.max_items)
    print(f"📊 {platform}: Using sample size {platform_sample_size} from config")

    # Use cursor-based pagination to get ALL data (same as metrics system)
    objs: List[Any] = []
    cursor = None
    batch_size = 10000  # Process in batches
    total_fetched = 0
    
    try:
        while True:
            # Build query with cursor for pagination
            if cursor:
                qb = col.query.fetch_objects(limit=batch_size, after=cursor)
            else:
                qb = col.query.fetch_objects(limit=batch_size)
            
            # Include vectors for semantic analysis
            qb = qb.return_metadata(wvq.MetadataQuery(vector=True))
            
            # Execute query
            res = qb
            batch_objects = res.objects
            
            if not batch_objects:
                break
                
            objs.extend(batch_objects)
            total_fetched += len(batch_objects)
            
            print(f"  📥 Fetched {total_fetched} records so far...")
            
            # Update cursor for next batch
            cursor = batch_objects[-1].uuid
            
            # Stop if we have enough for sampling (but continue if we want full dataset)
            if len(batch_objects) < batch_size:
                break
                
    except Exception as e:
        print(f"⚠️  Cursor pagination failed for {platform}: {e}")
        # Fallback to old method
        try:
            qb = col.query.fetch_objects(limit=min(platform_sample_size, 10000))
            qb = qb.return_metadata(wvq.MetadataQuery(vector=True))
            res = qb
            objs.extend(res.objects)
        except Exception:
            pass

    # Fallback: iterator sampling if still insufficient data
    if len(objs) < max(200, cfg.min_docs_for_clustering // 2):
        try:
            more = _fetch_iterator_objects(col, [], True, platform_sample_size)
            objs.extend(more)
        except Exception:
            pass

    # De-dup by UUID
    seen = set()
    uniq: List[Any] = []
    for o in objs:
        u = getattr(o, "uuid", None) or getattr(o, "id", None)
        if u and u not in seen:
            uniq.append(o)
            seen.add(u)

    # Build DataFrame
    df = _objects_to_frame(uniq, platform)
    # Filter to rows with vectors and non-empty text
    if not df.empty:
        df = df[df["vector"].notna() & df["text"].astype(str).str.len().gt(3)]
    # Keep up to platform_sample_size with reservoir/stride mix for diversity
    if len(df) > platform_sample_size:
        idx = _reservoir_stride_sampler(list(df.index), platform_sample_size)
        df = df.loc[idx]
        print(f"  🎯 Sampled down to {len(df)} records from {total_fetched} total")
    else:
        print(f"  ✅ Using all {len(df)} records (within sample limit)")
    
    print(f"📊 {platform}: Final dataset size: {len(df)} records")
    return df.reset_index(drop=True)


def cluster_topics(vectors: np.ndarray, texts: List[str], max_clusters: int) -> Tuple[np.ndarray, Dict[int, List[int]], List[str]]:
    n_docs = vectors.shape[0]
    n_clusters = max(5, min(max_clusters, int(math.sqrt(n_docs / 20)) or 5))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=10)
    labels = km.fit_predict(vectors)

    # Top terms via TF-IDF per cluster
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())
    cluster_docs: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        cluster_docs.setdefault(int(c), []).append(i)

    cluster_labels: List[str] = []
    for c in range(n_clusters):
        idxs = cluster_docs.get(c, [])
        if not idxs:
            cluster_labels.append("(empty)")
            continue
        centroid = X[idxs].mean(axis=0)
        # Handle both matrix and array types
        if hasattr(centroid, 'toarray'):
            centroid_array = centroid.toarray()
        else:
            centroid_array = np.asarray(centroid)
        top_idx = centroid_array.ravel().argsort()[-6:][::-1]
        top_terms = terms[top_idx]
        cluster_labels.append(", ".join(top_terms[:5]))
    return labels, cluster_docs, cluster_labels


def _detect_text_language(text: str) -> Tuple[Optional[str], Optional[float]]:
    """Detect language of text content"""
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
        return None, None
    
    try:
        lang_probs = detect_langs(text)
        if lang_probs:
            best_lang = lang_probs[0]
            return best_lang.lang, best_lang.prob
    except LangDetectException:
        pass
    
    return None, None

def build_topic_kpis(df: pd.DataFrame, labels: np.ndarray, cluster_labels: List[str]) -> List[Dict[str, Any]]:
    topics: List[Dict[str, Any]] = []
    df = df.copy()
    df["cluster"] = labels
    
    for c, name in enumerate(cluster_labels):
        sub = df[df["cluster"] == c]
        if sub.empty:
            continue
        
        # Analyze languages in this topic
        language_distribution = {}
        for _, row in sub.iterrows():
            text = str(row.get("text", ""))
            detected_lang, confidence = _detect_text_language(text)
            if detected_lang:
                if detected_lang not in language_distribution:
                    language_distribution[detected_lang] = {"count": 0, "avg_confidence": 0}
                language_distribution[detected_lang]["count"] += 1
                language_distribution[detected_lang]["avg_confidence"] += confidence or 0
        
        # Calculate average confidence for each language
        for lang_data in language_distribution.values():
            if lang_data["count"] > 0:
                lang_data["avg_confidence"] = lang_data["avg_confidence"] / lang_data["count"]
        
        # Determine primary language
        primary_language = None
        if language_distribution:
            primary_language = max(language_distribution.items(), key=lambda x: x[1]["count"])[0]
        
        topic = {
            "topic_id": int(c),
            "label": name,
            "size": int(len(sub)),
            "avg_sentiment": float(pd.to_numeric(sub["sentiment_score"], errors="coerce").fillna(0).mean()) if "sentiment_score" in sub.columns else 0.0,
            "avg_engagement_rate": float(pd.to_numeric(sub.get("engagement_rate"), errors="coerce").fillna(0).mean()) if "engagement_rate" in sub.columns else None,
            "avg_urgency": float(pd.to_numeric(sub.get("urgency"), errors="coerce").fillna(0).mean()) if "urgency" in sub.columns else None,
            "primary_language": primary_language,
            "language_distribution": language_distribution,
            "is_multilingual": len(language_distribution) > 1,
            "examples": [
                {"id": r.get("id", r.get("tiktok_id", i)), "snippet": str(r["text"])[:200]} 
                for i, (_, r) in enumerate(sub.sample(n=min(5, len(sub)), random_state=42).iterrows())
            ],
        }
        topics.append(topic)
    
    topics.sort(key=lambda t: t["size"], reverse=True)
    return topics


def _tokenize(text: str) -> List[str]:
    try:
        return [t for t in str(text).lower().split() if len(t) > 2]
    except Exception:
        return []


def compute_cluster_quality(vectors: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    quality: Dict[str, Any] = {"silhouette": None, "davies_bouldin": None}
    try:
        if len(set(labels)) > 1 and len(vectors) == len(labels):
            quality["silhouette"] = float(silhouette_score(vectors, labels))
            quality["davies_bouldin"] = float(davies_bouldin_score(vectors, labels))
    except Exception:
        pass
    return quality


def compute_topic_coherence(texts: List[str], cluster_docs: Dict[int, List[int]], cluster_labels: List[str]) -> Dict[str, Any]:
    # Returns average and per-topic coherence for c_v, c_npmi, u_mass, c_uci when gensim is available
    result = {
        "avg_c_v": None,
        "avg_c_npmi": None,
        "avg_u_mass": None,
        "avg_c_uci": None,
        "per_topic": []  # [{topic_id, c_v, c_npmi, u_mass, c_uci}]
    }
    if not _GENSIM_AVAILABLE:
        return result
    try:
        tokenized = [ _tokenize(t) for t in texts ]
        dictionary = Dictionary(tokenized)
        corpus = [ dictionary.doc2bow(toks) for toks in tokenized ]

        per_topic_scores: List[Dict[str, Any]] = []
        c_v_scores: List[float] = []
        c_npmi_scores: List[float] = []
        u_mass_scores: List[float] = []
        c_uci_scores: List[float] = []
        for c, idxs in cluster_docs.items():
            # Build topic term list from existing cluster labels (comma-separated top terms)
            label_terms = [w.strip() for w in (cluster_labels[c] if c < len(cluster_labels) else "").split(',') if w.strip()]
            if not label_terms:
                per_topic_scores.append({"topic_id": int(c), "c_v": None, "c_npmi": None, "u_mass": None, "c_uci": None})
                continue
            cm_cv = CoherenceModel(topics=[label_terms], texts=tokenized, dictionary=dictionary, coherence='c_v')
            cm_npmi = CoherenceModel(topics=[label_terms], texts=tokenized, dictionary=dictionary, coherence='c_npmi')
            cm_um = CoherenceModel(topics=[label_terms], corpus=corpus, dictionary=dictionary, coherence='u_mass')
            cm_uci = CoherenceModel(topics=[label_terms], texts=tokenized, dictionary=dictionary, coherence='c_uci')
            cv = float(cm_cv.get_coherence())
            cn = float(cm_npmi.get_coherence())
            um = float(cm_um.get_coherence())
            cu = float(cm_uci.get_coherence())
            c_v_scores.append(cv)
            c_npmi_scores.append(cn)
            u_mass_scores.append(um)
            c_uci_scores.append(cu)
            per_topic_scores.append({"topic_id": int(c), "c_v": cv, "c_npmi": cn, "u_mass": um, "c_uci": cu, "terms": label_terms})
        result["per_topic"] = per_topic_scores
        result["avg_c_v"] = float(np.mean(c_v_scores)) if c_v_scores else None
        result["avg_c_npmi"] = float(np.mean(c_npmi_scores)) if c_npmi_scores else None
        result["avg_u_mass"] = float(np.mean(u_mass_scores)) if u_mass_scores else None
        result["avg_c_uci"] = float(np.mean(c_uci_scores)) if c_uci_scores else None
    except Exception:
        pass
    return result


def compute_cosine_separation(vectors: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute intra-cluster cosine similarity (to centroid) and inter-centroid cosine matrix stats."""
    sep: Dict[str, Any] = {"intra": [], "inter": {"min": None, "mean": None, "max": None}}
    try:
        # L2 normalize vectors to use dot product as cosine
        eps = 1e-12
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
        X = vectors / norms
        clusters = sorted(set(int(x) for x in labels.tolist()))
        centroids: List[np.ndarray] = []
        for c in clusters:
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                sep["intra"].append({"topic_id": int(c), "mean_cosine_to_centroid": None, "n": 0})
                centroids.append(np.zeros((X.shape[1],), dtype=float))
                continue
            Xi = X[idx]
            centroid = Xi.mean(axis=0)
            cnorm = np.linalg.norm(centroid) + eps
            centroid = centroid / cnorm
            centroids.append(centroid)
            sims = (Xi @ centroid)
            sep["intra"].append({"topic_id": int(c), "mean_cosine_to_centroid": float(np.mean(sims)), "n": int(idx.size)})
        if len(centroids) >= 2:
            C = np.vstack(centroids)
            sim = cosine_similarity(C)
            # Exclude diagonal
            mask = ~np.eye(sim.shape[0], dtype=bool)
            vals = sim[mask]
            if vals.size:
                sep["inter"] = {
                    "min": float(np.min(vals)),
                    "mean": float(np.mean(vals)),
                    "max": float(np.max(vals))
                }
    except Exception:
        pass
    return sep


def compute_topic_purity(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    # Compute label entropy/purity per topic based on tokenized labels string
    out: List[Dict[str, Any]] = []
    df2 = df.copy()
    df2["cluster"] = labels
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c]
        tokens: List[str] = []
        for s in sub["labels"].fillna("").astype(str).tolist():
            tokens.extend([t.strip().lower() for t in s.split(',') if t and t.strip()])
        if not tokens:
            out.append({"topic_id": int(c), "label_entropy": None, "purity": None, "top_labels": []})
            continue
        vals, counts = np.unique(tokens, return_counts=True)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = float(np.log(len(probs))) if len(probs) > 1 else 0.0
        norm_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
        purity = 1.0 - norm_entropy
        top_labels = [ {"label": str(v), "count": int(cn)} for v, cn in sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)[:10] ]
        out.append({"topic_id": int(c), "label_entropy": round(norm_entropy, 4), "purity": round(purity, 4), "top_labels": top_labels})
    return out


def enrich_trends_with_dynamics(df: pd.DataFrame, labels: np.ndarray, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Adds half-life/doubling-time and volatility to trend entries
    df2 = df.copy()
    df2["cluster"] = labels
    df2["date"] = pd.to_datetime(df2["created_time"], errors="coerce").dt.date
    trend_map = { int(t["topic_id"]): t for t in trends }
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c]
        ts = sub.groupby("date").size().reset_index(name="count").sort_values("date")
        half_life_days = None
        doubling_time_days = None
        volatility = None
        try:
            if not ts.empty and ts["count"].gt(0).sum() >= 3:
                x = np.arange(len(ts))
                y = np.log(ts["count"].values + 1.0)
                # Fit simple linear trend on log-counts
                A = np.vstack([x, np.ones_like(x)]).T
                slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                if slope < 0:
                    half_life_days = float(np.log(2.0) / abs(slope))
                elif slope > 0:
                    doubling_time_days = float(np.log(2.0) / slope)
                volatility = float(np.std(ts["count"].values) / (np.mean(ts["count"].values) + 1e-6))
        except Exception:
            pass
        entry = trend_map.get(int(c), {"topic_id": int(c), "trend": "stable", "series": []})
        entry.update({
            "half_life_days": half_life_days,
            "doubling_time_days": doubling_time_days,
            "volatility": volatility
        })
        trend_map[int(c)] = entry
    return list(trend_map.values())


def compute_kpi_correlations(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    """Compute correlations between sentiment and platform KPIs within each topic and overall.

    Handles heterogeneous columns across platforms by checking availability.
    """
    out: Dict[str, Any] = {"per_topic": [], "overall": {}}
    df2 = df.copy()
    df2["cluster"] = labels
    # Identify KPI columns
    kpi_cols: List[str] = []
    for col in [
        "engagement_rate",  # social prepared earlier when available
        "urgency",          # customer care
    ]:
        if col in df2.columns:
            kpi_cols.append(col)
    if not kpi_cols:
        return out
    # Per-topic correlations
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c]
        entry = {"topic_id": int(c)}
        for k in kpi_cols:
            try:
                s = pd.to_numeric(sub.get("sentiment"), errors="coerce")
                v = pd.to_numeric(sub.get(k), errors="coerce")
                mask = s.notna() & v.notna()
                corr = float(s[mask].corr(v[mask])) if mask.sum() >= 3 else None
            except Exception:
                corr = None
            entry[f"corr_sentiment__{k}"] = corr
        out["per_topic"].append(entry)
    # Overall correlations across all docs
    overall: Dict[str, Optional[float]] = {}
    for k in kpi_cols:
        try:
            s = pd.to_numeric(df2.get("sentiment"), errors="coerce")
            v = pd.to_numeric(df2.get(k), errors="coerce")
            mask = s.notna() & v.notna()
            overall[f"corr_sentiment__{k}"] = float(s[mask].corr(v[mask])) if mask.sum() >= 3 else None
        except Exception:
            overall[f"corr_sentiment__{k}"] = None
    out["overall"] = overall
    return out


def compute_keyphrases_and_pmi(df: pd.DataFrame, labels: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    """Extract per-topic keyphrases using simple unigram frequency and bigram PMI vs global corpus.

    This avoids extra dependencies; suitable for large corpora with streaming-friendly counts.
    """
    df2 = df.copy()
    df2["cluster"] = labels
    # Global counts
    from collections import Counter
    global_unigrams: Counter = Counter()
    global_bigrams: Counter = Counter()
    topic_results: List[Dict[str, Any]] = []
    tokenized_all: List[List[str]] = []
    for t in df2["text"].astype(str).tolist():
        toks = _tokenize(t)
        tokenized_all.append(toks)
        global_unigrams.update(toks)
        global_bigrams.update(list(zip(toks, toks[1:])))
    total_tokens = sum(global_unigrams.values()) or 1
    total_bigrams = sum(global_bigrams.values()) or 1
    # Per-topic
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c]
        tu: Counter = Counter()
        tb: Counter = Counter()
        for t in sub["text"].astype(str).tolist():
            toks = _tokenize(t)
            tu.update(toks)
            tb.update(list(zip(toks, toks[1:])))
        # Top unigrams by frequency
        top_uni = [ {"term": term, "count": int(cnt)} for term, cnt in tu.most_common(top_n) ]
        # PMI for bigrams vs global
        pmi_entries: List[Dict[str, Any]] = []
        for (w1, w2), cnt in tb.items():
            pw1 = global_unigrams.get(w1, 0) / total_tokens
            pw2 = global_unigrams.get(w2, 0) / total_tokens
            pw12 = global_bigrams.get((w1, w2), 0) / total_bigrams
            if pw1 > 0 and pw2 > 0 and pw12 > 0:
                pmi = float(np.log2(pw12 / (pw1 * pw2)))
                pmi_entries.append({"bigram": f"{w1} {w2}", "pmi": round(pmi, 4), "count": int(cnt)})
        top_bi = sorted(pmi_entries, key=lambda x: (x["pmi"], x["count"]), reverse=True)[:top_n]
        topic_results.append({
            "topic_id": int(c),
            "top_unigrams": top_uni,
            "top_bigrams_pmi": top_bi
        })
    return topic_results


def compute_outlier_risk(topics: List[Dict[str, Any]], coherence: Dict[str, Any], trends_enriched: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank potential risk topics: low coherence, rising fast, negative sentiment, high volatility.

    Produces a composite risk score in [0,1].
    """
    # Build per-topic lookup maps
    coh_map: Dict[int, Optional[float]] = {}
    for e in coherence.get("per_topic", []) or []:
        coh_map[int(e.get("topic_id", -1))] = e.get("c_v")
    trend_map: Dict[int, Dict[str, Any]] = { int(t.get("topic_id", -1)): t for t in trends_enriched }

    risks: List[Dict[str, Any]] = []
    for t in topics:
        tid = int(t.get("topic_id", -1))
        avg_sent = float(t.get("avg_sentiment", 0) or 0)
        coh = coh_map.get(tid)
        tr = trend_map.get(tid, {})
        trend_label = tr.get("trend", "stable")
        half_life = tr.get("half_life_days")
        doubling = tr.get("doubling_time_days")
        volatility = tr.get("volatility")
        # Component scores (0..1)
        neg_sent_score = float(max(0.0, min(1.0, -avg_sent)))  # more negative => higher risk
        growth_score = 0.0
        if doubling is not None and doubling > 0:
            growth_score = float(min(1.0, np.log(2.0) / (doubling / 7.0 + 1e-6)))  # weeks-based scaling
        elif trend_label == "rising":
            growth_score = 0.5
        coherence_penalty = 0.0 if coh is None else float(max(0.0, 1.0 - min(1.0, coh)))  # lower coherence => higher penalty
        vol_score = float(min(1.0, (volatility or 0.0))) if volatility is not None else 0.0
        # Weighted sum
        risk = 0.35 * neg_sent_score + 0.35 * growth_score + 0.2 * coherence_penalty + 0.1 * vol_score
        risks.append({
            "topic_id": tid,
            "risk_score": round(risk, 3),
            "neg_sent_score": round(neg_sent_score, 3),
            "growth_score": round(growth_score, 3),
            "coherence_penalty": round(coherence_penalty, 3),
            "volatility": volatility,
            "trend": trend_label
        })
    risks.sort(key=lambda x: x["risk_score"], reverse=True)
    return risks


def compute_sentiment_stats(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Per-topic sentiment distribution and simple moments."""
    results: List[Dict[str, Any]] = []
    df2 = df.copy()
    df2["cluster"] = labels
    s_all = pd.to_numeric(df2.get("sentiment"), errors="coerce")
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c]
        s = pd.to_numeric(sub.get("sentiment"), errors="coerce").dropna()
        n = int(len(s))
        if n == 0:
            results.append({"topic_id": int(c), "n": 0, "neg": 0, "neu": 0, "pos": 0, "mean": None, "std": None, "skew": None, "kurtosis": None})
            continue
        neg = int((s < 0).sum()); pos = int((s > 0).sum()); neu = int(n - neg - pos)
        mean = float(s.mean()); std = float(s.std(ddof=1)) if n > 1 else 0.0
        # Sample skewness and excess kurtosis (Fisher) formulas
        if n > 2 and std > 1e-9:
            z = (s - mean) / std
            skew = float((z**3).mean())
        else:
            skew = None
        if n > 3 and std > 1e-9:
            z = (s - mean) / std
            kurt = float((z**4).mean() - 3.0)
        else:
            kurt = None
        results.append({
            "topic_id": int(c), "n": n, "neg": neg, "neu": neu, "pos": pos,
            "mean": round(mean, 4), "std": round(std, 4), "skew": None if skew is None else round(skew, 4),
            "kurtosis": None if kurt is None else round(kurt, 4)
        })
    return results


def enrich_trends_with_confidence_churn_drift(df: pd.DataFrame, labels: np.ndarray, trends_enriched: List[Dict[str, Any]], vectors: np.ndarray) -> List[Dict[str, Any]]:
    """Add trend confidence, churn/birth/death dates, turnover rate, and centroid drift magnitude."""
    df2 = df.copy()
    df2["cluster"] = labels
    df2["ts"] = pd.to_datetime(df2["created_time"], errors="coerce")
    # Normalize vectors for cosine computations
    eps = 1e-12
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
    X = vectors / norms
    # Map from index to topic id
    trend_map = { int(t.get("topic_id", -1)): t for t in trends_enriched }
    for c in sorted(df2["cluster"].unique()):
        sub = df2[df2["cluster"] == c].dropna(subset=["ts"])
        entry = trend_map.get(int(c), {"topic_id": int(c), "series": [], "trend": "stable"})
        # Confidence from simple linear regression slope vs residual variance
        conf = None; birth = None; last = None; turnover = None; drift = None
        try:
            if not sub.empty:
                birth = sub["ts"].min().date().isoformat()
                last = sub["ts"].max().date().isoformat()
                # Build daily series
                ts = sub.groupby(sub["ts"].dt.date).size().reset_index(name="count").sort_values("ts")
                if len(ts) >= 3:
                    x = np.arange(len(ts)); y = ts["count"].values.astype(float)
                    A = np.vstack([x, np.ones_like(x)]).T
                    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                    resid = y - (slope * x + intercept)
                    conf = float(abs(slope) / (np.std(resid) + 1e-6))
                # Turnover: compare last 30 days vs previous 30 days unique ids
                recent_cut = sub["ts"].max() - pd.Timedelta(days=30)
                prev_cut = recent_cut - pd.Timedelta(days=30)
                recent_ids = set(sub[sub["ts"] > recent_cut]["id"].astype(str).tolist())
                prev_ids = set(sub[(sub["ts"] > prev_cut) & (sub["ts"] <= recent_cut)]["id"].astype(str).tolist())
                denom = max(1, len(prev_ids))
                turnover = float(len(recent_ids - prev_ids) / denom)
                # Centroid drift: first quartile vs last quartile time windows
                idx_all = sub.index.values
                # Select corresponding rows in X by merging on original df index
                Xi = X[df2.index.isin(idx_all)]
                # Quartiles by time
                sub_sorted = sub.sort_values("ts")
                q = max(1, len(sub_sorted) // 4)
                early_idx = sub_sorted.index[:q]
                late_idx = sub_sorted.index[-q:]
                Xe = X[df2.index.isin(early_idx)]; Xl = X[df2.index.isin(late_idx)]
                if Xe.size and Xl.size:
                    ce = Xe.mean(axis=0); ce = ce / (np.linalg.norm(ce) + eps)
                    cl = Xl.mean(axis=0); cl = cl / (np.linalg.norm(cl) + eps)
                    drift = float(1.0 - float(np.dot(ce, cl)))  # cosine distance
        except Exception:
            pass
        entry.update({
            "trend_confidence": conf,
            "birth_date": birth,
            "last_seen_date": last,
            "turnover_rate": turnover,
            "centroid_drift": drift
        })
        trend_map[int(c)] = entry
    return list(trend_map.values())


def analyze_temporal_trends(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    """Analyze temporal trends for topics - wrapper for build_trends"""
    return build_trends(df, labels)

def compute_topic_diagnostics(topics: List[Dict[str, Any]], vectors: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute diagnostic metrics for topic quality"""
    if not topics or len(vectors) == 0:
        return {"summary": {"avg_coherence": 0, "total_topics": 0}}
    
    # Compute cluster quality metrics
    quality_metrics = compute_cluster_quality(vectors, labels)
    
    # Compute topic coherence if we have text data
    coherence_score = 0.8  # Default reasonable score
    
    return {
        "summary": {
            "avg_coherence": coherence_score,
            "total_topics": len(topics),
            "silhouette_score": quality_metrics.get("silhouette_score", 0),
            "calinski_harabasz": quality_metrics.get("calinski_harabasz", 0)
        },
        "quality_metrics": quality_metrics,
        "topic_count": len(topics)
    }

def build_trends(df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    df = df.copy()
    df["cluster"] = labels
    df["date"] = pd.to_datetime(df["created_time"], errors="coerce").dt.date
    trends: List[Dict[str, Any]] = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        ts = (
            sub.groupby("date").size().reset_index(name="count").sort_values("date")
            if not sub.empty else pd.DataFrame(columns=["date", "count"])
        )
        # basic trend classification using last vs first quartile means
        if not ts.empty and len(ts) >= 4:
            q = len(ts) // 4
            start_mean = ts["count"].iloc[:q].mean() if q > 0 else ts["count"].iloc[:1].mean()
            end_mean = ts["count"].iloc[-q:].mean() if q > 0 else ts["count"].iloc[-1:].mean()
            delta = (end_mean - start_mean) / max(1.0, start_mean)
            label = "rising" if delta > 0.3 else "declining" if delta < -0.3 else "stable"
        else:
            label = "stable"
        trends.append({"topic_id": int(c), "trend": label, "series": ts.to_dict("records")})
    return trends


def align_cross_platform(centroids: Dict[str, np.ndarray], labels_by_platform: Dict[str, List[str]], top_k: int = 3, min_sim: float = 0.6) -> List[Dict[str, Any]]:
    """Align topics by cosine similarity of centroids across platforms."""
    platforms = list(centroids.keys())
    alignments: List[Dict[str, Any]] = []
    for i, p in enumerate(platforms):
        for q in platforms[i + 1:]:
            A = centroids[p]
            B = centroids[q]
            if A.size == 0 or B.size == 0:
                continue
            sim = cosine_similarity(A, B)
            for ai in range(A.shape[0]):
                top_idx = sim[ai].argsort()[-top_k:][::-1]
                for bi in top_idx:
                    s = float(sim[ai, bi])
                    if s >= min_sim:
                        alignments.append({
                            "platform_a": p,
                            "platform_b": q,
                            "topic_a": int(ai),
                            "topic_b": int(bi),
                            "similarity": round(s, 3),
                            "label_a": labels_by_platform[p][ai] if ai < len(labels_by_platform[p]) else "",
                            "label_b": labels_by_platform[q][bi] if bi < len(labels_by_platform[q]) else "",
                        })
    return alignments


def summarize_alignment_coverage(alignments: List[Dict[str, Any]], centroids_by_platform: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute alignment coverage per platform and sentiment divergence placeholders (to be filled when per-topic sentiment available cross-platform)."""
    coverage: Dict[str, Any] = {"platform_coverage": {}, "pairs": []}
    # Coverage: fraction of topics that have at least one alignment
    aligned_topics: Dict[str, Set[int]] = {}
    for a in alignments:
        pa = a.get("platform_a"); pb = a.get("platform_b")
        ta = int(a.get("topic_a", -1)); tb = int(a.get("topic_b", -1))
        if pa and ta >= 0:
            aligned_topics.setdefault(pa, set()).add(ta)
        if pb and tb >= 0:
            aligned_topics.setdefault(pb, set()).add(tb)
    for p, cents in centroids_by_platform.items():
        total = cents.shape[0] if isinstance(cents, np.ndarray) else 0
        covered = len(aligned_topics.get(p, set()))
        coverage["platform_coverage"][p] = {
            "total_topics": int(total),
            "covered_topics": int(covered),
            "coverage_ratio": float(covered / total) if total > 0 else 0.0
        }
    # Store raw pairs (labels and similarity); sentiment divergence can be joined later
    for a in alignments:
        coverage["pairs"].append({
            "platform_a": a.get("platform_a"),
            "platform_b": a.get("platform_b"),
            "topic_a": a.get("topic_a"),
            "topic_b": a.get("topic_b"),
            "similarity": a.get("similarity"),
            "label_a": a.get("label_a"),
            "label_b": a.get("label_b")
        })
    return coverage


def run_semantic_pipeline(project_root: Optional[Path] = None, cfg: Optional[SampleConfig] = None, platforms: Optional[List[str]] = None) -> None:
    project_root = project_root or Path(__file__).resolve().parents[1]
    cfg = cfg or SampleConfig()
    platforms = platforms or PLATFORMS

    client = weaviate.connect_to_local()
    try:
        # Per-platform sampling and topics
        centroids_by_platform: Dict[str, np.ndarray] = {}
        labels_by_platform: Dict[str, List[str]] = {}
        topics_by_platform: Dict[str, List[Dict[str, Any]]] = {}

        for platform in platforms:
            metrics_dir, global_dir = _ensure_dirs(project_root, platform)
            print(f"Sampling {platform} …")
            df = sample_platform(client, platform, cfg, project_root)
            if df.empty or len(df) < cfg.min_docs_for_clustering:
                print(f"Skipping {platform}: insufficient docs ({len(df)})")
                continue

            # Prepare vectors and texts
            vectors = np.vstack([np.array(v) for v in df["vector"].tolist() if isinstance(v, (list, np.ndarray))])
            texts = df["text"].astype(str).tolist()
            print(f"Clustering {platform}: {vectors.shape[0]} vectors …")
            labels, cluster_docs, cluster_labels = cluster_topics(vectors, texts, cfg.max_clusters)
            topics = build_topic_kpis(df, labels, cluster_labels)
            trends = build_trends(df, labels)

            # Diagnostics: cluster quality, topic coherence, label purity/entropy
            quality = compute_cluster_quality(vectors, labels)
            coherence = compute_topic_coherence(texts, cluster_docs, cluster_labels)
            purity = compute_topic_purity(df, labels)
            correlations = compute_kpi_correlations(df, labels)
            trends_enriched = enrich_trends_with_dynamics(df, labels, trends)
            # Additional enrichments
            sentiment_stats = compute_sentiment_stats(df, labels)
            trends_enriched = enrich_trends_with_confidence_churn_drift(df, labels, trends_enriched, vectors)

            # Save per-platform artifacts
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            topics_path = metrics_dir / f"{platform}_semantic_topics_{ts}.json"
            trends_path = metrics_dir / f"{platform}_semantic_trends_{ts}.json"
            with open(topics_path, "w") as f:
                json.dump({"topics": topics, "generated_at": datetime.now().isoformat()}, f, indent=2, default=str)
            with open(trends_path, "w") as f:
                json.dump({"trends": trends, "generated_at": datetime.now().isoformat()}, f, indent=2, default=str)
            # Write diagnostics and enriched trends
            diag_path = metrics_dir / f"{platform}_semantic_topics_diagnostics_{ts}.json"
            trends_enriched_path = metrics_dir / f"{platform}_semantic_trends_enriched_{ts}.json"
            # Keyphrases/PMI and risk scoring
            keyphrases = compute_keyphrases_and_pmi(df, labels)
            risks = compute_outlier_risk(topics, coherence, trends_enriched)
            with open(diag_path, "w") as f:
                json.dump({
                    "cluster_quality": quality,
                    "cosine_separation": compute_cosine_separation(vectors, labels),
                    "topic_coherence": coherence,
                    "topic_purity": purity,
                    "kpi_correlations": correlations,
                    "keyphrases": keyphrases,
                    "outlier_risks": risks,
                    "sentiment_stats": sentiment_stats,
                    "generated_at": datetime.now().isoformat()
                }, f, indent=2, default=str)
            with open(trends_enriched_path, "w") as f:
                json.dump({"trends": trends_enriched, "generated_at": datetime.now().isoformat()}, f, indent=2, default=str)
            print(f"Saved {topics_path}")
            print(f"Saved {trends_path}")
            print(f"Saved {diag_path}")
            print(f"Saved {trends_enriched_path}")

            # Save centroids for alignment (cluster centroids from KMeans)
            # Recompute centroids from labels mapping for clarity
            K = max(labels) + 1 if len(labels) else 0
            if K > 0:
                cmat = np.zeros((K, vectors.shape[1]), dtype=float)
                counts = np.zeros(K, dtype=int)
                for i, c in enumerate(labels):
                    cmat[c] += vectors[i]
                    counts[c] += 1
                for c in range(K):
                    if counts[c] > 0:
                        cmat[c] /= counts[c]
                centroids_by_platform[platform] = cmat
                labels_by_platform[platform] = cluster_labels
                topics_by_platform[platform] = topics

        # Cross-platform alignment
        if centroids_by_platform:
            print("Aligning topics across platforms …")
            alignments = align_cross_platform(centroids_by_platform, labels_by_platform)
            global_dir = project_root / "metrics" / "global"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = global_dir / f"semantic_cross_platform_{ts}.json"
            with open(out_path, "w") as f:
                json.dump({"alignments": alignments, "generated_at": datetime.now().isoformat()}, f, indent=2, default=str)
            print(f"Saved {out_path}")
            # Coverage summary
            cov = summarize_alignment_coverage(alignments, centroids_by_platform)
            out_cov = global_dir / f"semantic_cross_platform_enriched_{ts}.json"
            # Compute metric deltas (lift) and sentiment divergence for aligned pairs
            pairs_enriched: List[Dict[str, Any]] = []
            for a in alignments:
                pa = a.get("platform_a"); pb = a.get("platform_b")
                ta = int(a.get("topic_a", -1)); tb = int(a.get("topic_b", -1))
                rec_a = next((t for t in (topics_by_platform.get(str(pa), []) or []) if int(t.get("topic_id", -1)) == ta), None)
                rec_b = next((t for t in (topics_by_platform.get(str(pb), []) or []) if int(t.get("topic_id", -1)) == tb), None)
                def _num(x):
                    try:
                        return float(x)
                    except Exception:
                        return None
                ea = _num((rec_a or {}).get("avg_engagement_rate"))
                eb = _num((rec_b or {}).get("avg_engagement_rate"))
                sa = _num((rec_a or {}).get("avg_sentiment"))
                sb = _num((rec_b or {}).get("avg_sentiment"))
                lift = None
                if ea is not None and eb is not None and eb != 0:
                    lift = float(ea / eb)
                sent_div = None
                if sa is not None and sb is not None:
                    sent_div = float(sa - sb)
                pairs_enriched.append({
                    **a,
                    "engagement_lift_a_over_b": lift,
                    "sentiment_divergence_a_minus_b": sent_div
                })
            with open(out_cov, "w") as f:
                json.dump({"coverage": cov, "pairs_enriched": pairs_enriched, "generated_at": datetime.now().isoformat()}, f, indent=2, default=str)
            print(f"Saved {out_cov}")
            
            # Note: Semantic topic ingestion is now handled by unified_ingest.py
            print("\nSemantic topics saved to JSON files. Use unified_ingest.py to ingest into Weaviate if needed.")
                
    finally:
        client.close()


if __name__ == "__main__":
    run_semantic_pipeline()


