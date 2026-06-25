import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROUNDS = [
    "round_0_cps",
    "round_1_random",
    "round_2_random",
    "round_3_random",
]

ROW_METRICS = ["exact_match_pct", "cosine_similarity", "bert_score_f1",
               "jaccard_similarity", "levenshtein_similarity"]


def main():
    rows = []
    for round_name in ROUNDS:
        csv_path = os.path.join(BASE_DIR, round_name, "assignment_parsed.csv")
        if not os.path.exists(csv_path):
            print(f"SKIP: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        row = {"round": round_name}
        for m in ROW_METRICS:
            row[m] = df[m].mean() if m in df.columns else None
        rows.append(row)

    out = pd.DataFrame(rows)

    clust_path = os.path.join(BASE_DIR, "clustering_metrics.csv")
    if os.path.exists(clust_path):
        clust = pd.read_csv(clust_path)
        out = out.merge(clust, on="round", how="left")

    out_path = os.path.join(BASE_DIR, "metrics_summary.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
