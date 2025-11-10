from pathlib import Path
import json, pandas as pd
import matplotlib.pyplot as plt

OUT = Path("offline_eval_results/out")
sumj = json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
df = pd.read_csv(OUT/"per_question.csv")

# Retrieval bar
plt.figure()
x = ["Recall@K","MRR","Top1.mean"]
y = [sumj["retrieval"]["Recall@K"], sumj["retrieval"]["MRR"], sumj["retrieval"]["Top1Score.mean"]]
plt.bar(x, y); plt.title("Retrieval Metrics"); plt.ylim(0,1)
plt.savefig(OUT/"retrieval_metrics.png", bbox_inches="tight")

# Top1 histogram
plt.figure()
df["top1_score"].dropna().plot(kind="hist", bins=20)
plt.title("Top-1 Score Distribution"); plt.xlabel("score"); plt.ylabel("count")
plt.savefig(OUT/"top1_hist.png", bbox_inches="tight")

# F1 by route box
plt.figure()
df.boxplot(column="f1", by="route")
plt.title("F1 by Route"); plt.suptitle(""); plt.ylim(0,1)
plt.savefig(OUT/"f1_by_route.png", bbox_inches="tight")
print("Saved PNGs in offline_eval_results/out")
