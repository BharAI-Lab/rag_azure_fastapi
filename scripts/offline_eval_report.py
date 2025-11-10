#!/usr/bin/env python3
"""
Build a standalone HTML report for offline evaluation results.
Reads:
  offline_eval_results/out/summary.json
  offline_eval_results/out/per_question.csv
Writes:
  offline_eval_results/out/report.html
"""

from pathlib import Path
import json
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "offline_eval_results" / "out"
SUMMARY = OUT / "summary.json"
DETAILS = OUT / "per_question.csv"
REPORT = OUT / "report.html"

def main():
    if not SUMMARY.exists() or not DETAILS.exists():
        raise SystemExit("Run offline_evaluate.py first to produce summary.json & per_question.csv")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    df = pd.read_csv(DETAILS)

    # ----- Figure 1: Retrieval metrics (bar)
    retr = summary["retrieval"]
    x = ["Recall@K", "MRR", "Top1Score.mean"]
    y = [retr.get("Recall@K", None), retr.get("MRR", None), retr.get("Top1Score.mean", None)]
    fig_retrieval = go.Figure(data=[go.Bar(x=x, y=y)])
    fig_retrieval.update_layout(title="Retrieval Metrics", yaxis_title="Value")

    # ----- Figure 2: Routing distribution (pie)
    routing = summary.get("routing", {})
    if routing:
        fig_routing = go.Figure(
            data=[go.Pie(labels=list(routing.keys()), values=[v*100 for v in routing.values()])]
        )
        fig_routing.update_layout(title="Routing Distribution (%)")
    else:
        fig_routing = go.Figure()
        fig_routing.update_layout(title="Routing Distribution (no data)")

    # ----- Figure 3: Top-1 score distribution (hist)
    if "top1_score" in df.columns and not df["top1_score"].isna().all():
        fig_top1 = px.histogram(df, x="top1_score", nbins=20, title="Top-1 Score Distribution")
    else:
        fig_top1 = go.Figure(); fig_top1.update_layout(title="Top-1 Score Distribution (no data)")

    # ----- Figure 4: F1 by route (box)
    if {"f1","route"}.issubset(df.columns) and not df["f1"].isna().all():
        fig_f1_route = px.box(df, x="route", y="f1", points="all", title="F1 by Route")
    else:
        fig_f1_route = go.Figure(); fig_f1_route.update_layout(title="F1 by Route (no data)")

    # ----- Figure 5: Latency histogram
    lat = summary.get("latency_sec", {})
    fig_latency = go.Figure()
    if "pred_answer" in df.columns:
        # if you logged per-sample latency in df, plot it; else fallback to summary bar
        pass
    # fallback to summary bar
    fig_latency = go.Figure(data=[go.Bar(x=["avg","p95"], y=[lat.get("avg", None), lat.get("p95", None)])])
    fig_latency.update_layout(title="Latency (seconds)")

    # ----- Small numeric cards
    def kv(k, v): return f"<div><b>{k}</b>: {v}</div>"
    cards_html = "".join([
        kv("N", summary["n"]),
        kv("Recall@K", retr.get("Recall@K")),
        kv("MRR", retr.get("MRR")),
        kv("Top1 mean / p50 / p90",
           f'{retr.get("Top1Score.mean")}, {retr.get("Top1Score.p50")}, {retr.get("Top1Score.p90")}'),
        kv("EM", summary["answering"].get("EM")),
        kv("F1", summary["answering"].get("F1")),
        kv("Groundedness (when grounded)", summary["answering"].get("Groundedness(when grounded)")),
        kv("Routing", summary.get("routing")),
        kv("Latency avg / p95 (s)", f'{lat.get("avg")}, {lat.get("p95")}'),
    ])

    # ----- Table preview (first 25)
    preview = df[["id","route","top1_score","recall_at_k","mrr","em","f1"]].head(25).to_html(index=False)

    # ----- Assemble HTML
    parts = [
        "<html><head><meta charset='utf-8'><title>RAG Offline Evaluation Report</title></head><body>",
        "<h1>RAG Offline Evaluation Report</h1>",
        f"<h3>Overview</h3><div style='display:flex;gap:24px;flex-wrap:wrap'>{cards_html}</div>",
        "<hr/>",
        fig_retrieval.to_html(full_html=False, include_plotlyjs='cdn'),
        fig_routing.to_html(full_html=False, include_plotlyjs=False),
        fig_top1.to_html(full_html=False, include_plotlyjs=False),
        fig_f1_route.to_html(full_html=False, include_plotlyjs=False),
        fig_latency.to_html(full_html=False, include_plotlyjs=False),
        "<hr/>",
        "<h3>Per-question (preview)</h3>",
        preview,
        f"<p>Full CSV: <code>{DETAILS}</code></p>",
        "</body></html>"
    ]
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {REPORT}")

if __name__ == "__main__":
    main()
