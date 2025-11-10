#!/usr/bin/env python3
"""
Offline statistical evaluation for the GDPR RAG Assistant.

This script evaluates your live RAG pipeline (from `app/rag.py`) against a gold
Q/A dataset and writes both a per-question table and an aggregate summary.

It is **purely offline** and does not modify your API. You can later expose the
generated summary via a small FastAPI endpoint if you want to view charts in
the UI.

-------------------------------------------------------------------------------
Directory layout (assumed)
-------------------------------------------------------------------------------
repo_root/
  app/
    rag.py
  dataset_files/
    gdpr_eval_qa.json
  offline_eval_results/
    out/
  scripts/
    offline_evaluate.py   <-- this file

-------------------------------------------------------------------------------
What it measures
-------------------------------------------------------------------------------
Retrieval:
  • Recall@K
  • MRR (Mean Reciprocal Rank)
  • Top-1 score (mean / p50 / p90)

Answering:
  • Exact Match (normalized)
  • F1 (normalized)
  • Groundedness (when routed as "grounded"): whether the gold span appears
    in any of the top-K docs or the returned source snippets.

Routing:
  • Distribution of Grounded / Hybrid / Off-topic
  • Hybrid triggers (if present)

Latency:
  • Average and p95 latency (seconds)

-------------------------------------------------------------------------------
Inputs
-------------------------------------------------------------------------------
• --eval-file
  Path to a JSON file of gold Q/A items (default: dataset_files/gdpr_eval_qa.json)
  Expected per-item fields:
    - id: str|int
    - question: str
    - answer: str           (gold answer)
    - answer_span: str      (optional; if missing, uses answer)

-------------------------------------------------------------------------------
Outputs
-------------------------------------------------------------------------------
Writes to --out-dir (default: offline_eval_results/out):
  • per_question.csv  (row-level metrics and predictions)
  • summary.json      (aggregate metrics)

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------
# From anywhere (CWD-agnostic)
python scripts/offline_evaluate.py \
  --eval-file dataset_files/gdpr_eval_qa.json \
  --out-dir offline_eval_results/out \
  --k 4

"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Optional, cast

import pandas as pd

# -----------------------------------------------------------------------------
# Path resolution (robust to CWD)
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent  # scripts/ -> repo root

# Ensure we can import "app.*" no matter where we run from
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# Import the live pipeline
# -----------------------------------------------------------------------------
try:
    # - rag_answer(query: str) -> tuple[str, list[dict]]
    # - vs: AzureSearch vector store with .similarity_search_with_relevance_scores(query, k)
    from app import rag as ragmod
    from app.rag import rag_answer
except Exception as e:
    raise SystemExit(f"Failed to import app.rag from {REPO_ROOT}/app: {e!r}")

VS = getattr(ragmod, "vs", None)
RETRIEVE_FN = getattr(ragmod, "_retrieve", None)
DEFAULT_TOP_K: int = int(getattr(ragmod, "TOP_K", 4))

# -----------------------------------------------------------------------------
# Typing aliases for hits
# -----------------------------------------------------------------------------
# A retrieval "hit" is (Document-like, score float). We don't assume the doc type.
Hit = Tuple[Any, float]
Hits = List[Hit]

# -----------------------------------------------------------------------------
# Text normalization helpers (for fair EM/F1)
# -----------------------------------------------------------------------------
MD_LABELS_PATTERN = re.compile(
    r"\*\*grounded answer.*?\*\*:\s*|\*\*general .*?\*\*:\s*|^your question is outside the scope of gdpr\.\s*",
    re.IGNORECASE | re.DOTALL,
)


def normalize_text(s: Optional[str]) -> str:
    """
    Normalize a string for robust lexical comparison.

    The normalization is intentionally conservative to:
      - remove markdown decorations / headers your pipeline adds
      - lowercase
      - strip punctuation except spaces
      - collapse whitespace

    Args:
        s: Input string (possibly None).

    Returns:
        A normalized string suitable for token-based EM/F1 comparisons.
    """
    if not s:
        return ""
    s = s.strip().lower()
    # Remove markdown labels/prefixes we add in answers
    s = MD_LABELS_PATTERN.sub("", s)
    # Strip common markdown punctuation and headers
    s = re.sub(r"[*_`>#]", " ", s)
    # Remove non-alnum (keep spaces)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: Optional[str]) -> List[str]:
    """
    Tokenize a string after normalization by splitting on word characters.

    Args:
        s: Raw input string.

    Returns:
        List of tokens (lowercased, punctuation-stripped).
    """
    return re.findall(r"\w+", normalize_text(s))


# -----------------------------------------------------------------------------
# Metric functions
# -----------------------------------------------------------------------------
def exact_match(pred: str, gold: str) -> int:
    """
    Compute normalized exact match (EM) between prediction and gold.

    Args:
        pred: Model answer.
        gold: Gold/target answer.

    Returns:
        1 if normalized strings match exactly; 0 otherwise.
    """
    return int(normalize_text(pred) == normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    """
    Compute token-level F1 between prediction and gold (normalized).

    Args:
        pred: Model answer.
        gold: Gold/target answer.

    Returns:
        F1 score in [0.0, 1.0].
    """
    p, g = Counter(tokenize(pred)), Counter(tokenize(gold))
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    precision = common / max(1, sum(p.values()))
    recall = common / max(1, sum(g.values()))
    return 2 * precision * recall / (precision + recall)


def _probe_span(gold_span: str) -> str:
    """
    Build a short normalized probe used to search within retrieved contents/snippets.

    Args:
        gold_span: Gold span string (or full answer if span not provided).

    Returns:
        A short normalized prefix (first ~50 chars) to probe within hits.
    """
    return normalize_text(gold_span)[:50]


def recall_at_k(hits: Sequence[Hit], gold_span: str) -> int:
    """
    Compute Recall@K: is the gold span present in any of the top-K documents?

    Args:
        hits: Sequence of (doc-like, score) retrieval hits.
        gold_span: Gold span string (or full answer if span not provided).

    Returns:
        1 if the gold span appears in any top-K doc content; 0 otherwise.
    """
    if not hits:
        return 0
    probe = _probe_span(gold_span)
    for doc, _score in hits:
        content = normalize_text(getattr(doc, "page_content", "") or "")
        if probe and probe in content:
            return 1
    return 0


def mrr(hits: Sequence[Hit], gold_span: str) -> float:
    """
    Compute Mean Reciprocal Rank for a single query.

    The reciprocal rank is 1/rank of the first hit that contains the gold span.
    If no hit contains it, the RR is 0. MRR is the mean of RR over all queries
    (aggregation is done outside this function).

    Args:
        hits: Sequence of (doc-like, score) retrieval hits.
        gold_span: Gold span string (or full answer if span not provided).

    Returns:
        Reciprocal rank for this query in [0.0, 1.0].
    """
    if not hits:
        return 0.0
    probe = _probe_span(gold_span)
    for i, (doc, _score) in enumerate(hits, start=1):
        content = normalize_text(getattr(doc, "page_content", "") or "")
        if probe and probe in content:
            return 1.0 / i
    return 0.0


# -----------------------------------------------------------------------------
# Routing detection (matches our app/rag.py prefixes)
# -----------------------------------------------------------------------------
def route_from_answer(answer: str) -> str:
    """
    Infer which branch the pipeline used by inspecting the answer prefix.

    Returns:
        "grounded" | "offtopic" | "hybrid"
    """
    a = (answer or "").strip().lower()
    if a.startswith("**grounded answer (gdpr dataset):**".lower()):
        return "grounded"
    if a.startswith("your question is outside the scope of gdpr.".lower()):
        return "offtopic"
    return "hybrid"


def hybrid_reason(answer: str) -> Optional[str]:
    """
    Provide a coarse reason for Hybrid route if detectable from the prefix.

    Returns:
        "no_context_or_weak" if we see the explicit 'no direct answer...' line,
        else None.
    """
    a = (answer or "").strip().lower()
    if a.startswith("no direct answer found in the gdpr dataset.".lower()):
        return "no_context_or_weak"
    return None


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Entry point: run the offline statistical evaluation.

    CLI Args:
        --eval-file: path to the gold Q/A JSON file (default:
                     dataset_files/gdpr_eval_qa.json)
        --k:         retrieval depth K for Recall@K/MRR (default: app.rag.TOP_K or 4)
        --limit:     limit number of examples (0 = all)
        --out-dir:   output directory (default: offline_eval_results/out)

    Side Effects:
        Writes CSV/JSON under out-dir and prints the summary JSON to stdout.
    """
    parser = argparse.ArgumentParser(description="Offline statistical evaluation for GDPR RAG.")
    parser.add_argument(
        "--eval-file",
        default=str(REPO_ROOT / "dataset_files" / "gdpr_eval_qa.json"),
        help="Path to gold Q/A json (default: dataset_files/gdpr_eval_qa.json)",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K,
        help=f"Top-K for retrieval (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of examples; 0 means all"
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "offline_eval_results" / "out"),
        help="Directory to write summary.json and per_question.csv "
             "(default: offline_eval_results/out)",
    )
    args = parser.parse_args()

    # Resolve IO paths
    eval_path = Path(args.eval_file).resolve()
    if not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read eval file {eval_path}: {e!r}")

    if args.limit and args.limit > 0:
        data = data[: args.limit]

    rows: List[dict] = []
    latencies: List[float] = []
    K = max(1, int(args.k))

    # Evaluate each item
    for item in data:
        q = item.get("question", "")
        gold = item.get("answer", "")
        gold_span = item.get("answer_span", gold)
        example_id = item.get("id", "")

        t0 = time.time()

        # Retrieval-only (typed)
        hits: Hits = []
        try:
            if VS is not None and hasattr(VS, "similarity_search_with_relevance_scores"):
                raw_hits: Any = VS.similarity_search_with_relevance_scores(q, k=K)
                hits = cast(Hits, raw_hits or [])
            elif callable(RETRIEVE_FN):
                raw_hits: Any = RETRIEVE_FN(q, K)
                hits = cast(Hits, raw_hits or [])
        except Exception:
            hits = []

        top1 = float(hits[0][1]) if hits else 0.0

        # Full pipeline answer
        try:
            pred_answer, sources = rag_answer(q)
        except Exception as e:
            pred_answer, sources = f"ERROR: {e}", []

        latencies.append(time.time() - t0)

        # Routing
        route = route_from_answer(pred_answer)
        hybrid_cause = hybrid_reason(pred_answer) if route == "hybrid" else None

        # Retrieval metrics
        r_at_k = recall_at_k(hits, gold_span)
        rr = mrr(hits, gold_span)

        # Answering metrics
        em = exact_match(pred_answer, gold)
        f1 = f1_score(pred_answer, gold)

        # Groundedness (only when grounded route)
        grounded_hit = 0
        if route == "grounded":
            probe = _probe_span(gold_span)
            in_sources = any(
                probe and probe in normalize_text((s or {}).get("snippet", ""))
                for s in (sources or [])
            )
            in_docs = any(
                probe and probe in normalize_text(getattr(d, "page_content", "") or "")
                for d, _ in (hits or [])
            )
            grounded_hit = int(in_sources or in_docs)

        rows.append({
            "id": example_id,
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "route": route,
            "hybrid_cause": hybrid_cause,
            "top1_score": top1,
            "recall_at_k": r_at_k,
            "mrr": rr,
            "em": em,
            "f1": f1,
            "groundedness_hit": grounded_hit,
        })

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Aggregate summaries
    def safe_mean(series: pd.Series) -> Optional[float]:
        """Return float(series.mean()) or None for empty series."""
        return float(series.mean()) if len(series) else None

    retrieval_summary = {
        "Recall@K": safe_mean(df["recall_at_k"]),
        "MRR": safe_mean(df["mrr"]),
        "Top1Score.mean": safe_mean(df["top1_score"]),
        "Top1Score.p50": float(df["top1_score"].median()) if len(df) else None,
        "Top1Score.p90": float(df["top1_score"].quantile(0.9)) if len(df) else None,
    }

    answering_summary = {
        "EM": safe_mean(df["em"]),
        "F1": safe_mean(df["f1"]),
        "Groundedness(when grounded)": safe_mean(
            df.loc[df["route"] == "grounded", "groundedness_hit"]
        ),
    }

    routing_dist = df["route"].value_counts(normalize=True).to_dict() if len(df) else {}
    hybrid_triggers = (
        df["hybrid_cause"].value_counts(dropna=True, normalize=True).to_dict()
        if "hybrid_cause" in df.columns and len(df)
        else {}
    )

    avg_latency = stats.mean(latencies) if latencies else None
    p95_latency = None
    if latencies and len(latencies) >= 20:
        # p95 via 20-quantiles → index 18 ≈ ~95th percentile
        p95_latency = stats.quantiles(latencies, n=20)[18]

    summary = {
        "n": len(df),
        "retrieval": retrieval_summary,
        "answering": answering_summary,
        "routing": routing_dist,
        "hybrid_triggers": hybrid_triggers,
        "latency_sec": {"avg": avg_latency, "p95": p95_latency},
    }

    # Write outputs
    per_q_path = Path(args.out_dir) / "per_question.csv"
    sum_path = Path(args.out_dir) / "summary.json"
    df.to_csv(per_q_path, index=False)
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print to console (useful in CI)
    print(json.dumps(summary, indent=2))

    # Best-effort cleanup (avoid finalizer noise at interpreter shutdown)
    try:
        setattr(ragmod, "vs", None)
    except Exception:
        pass


if __name__ == "__main__":
    main()
