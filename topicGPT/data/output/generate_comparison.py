import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTERING_METRICS = [
    "harmonic_mean_purity_p1",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "topic_coherence_cv",
    "topic_alignment_pn",
]

DATASETS = ["bills", "wiki"]


def main():
    for ds in DATASETS:
        ds_dir = os.path.join(BASE_DIR, ds)

        cps_rand_path = os.path.join(ds_dir, "cps_vs_random_summary.csv")
        bertopic_path = os.path.join(ds_dir, "bertopic_metrics.csv")

        if not os.path.exists(cps_rand_path) or not os.path.exists(bertopic_path):
            print(f"SKIP: {ds} — arquivos faltando")
            continue

        cps_rand = pd.read_csv(cps_rand_path)
        bt = pd.read_csv(bertopic_path)
        bt_mean = bt[bt["run"] == "mean"].iloc[0]
        bt_std = bt[bt["run"] == "std"].iloc[0]

        rows = []
        for m in CLUSTERING_METRICS:
            cr_row = cps_rand[cps_rand["metric"] == m]
            if cr_row.empty:
                continue
            rows.append({
                "metric": m,
                "BERTopic_mean": round(float(bt_mean[m]), 4),
                "BERTopic_std": round(float(bt_std[m]), 4),
                "CPS": round(float(cr_row["CPS"].values[0]), 4),
                "Random_mean": round(float(cr_row["random_mean"].values[0]), 4),
                "Random_std": round(float(cr_row["random_std"].values[0]), 4),
            })

        out = pd.DataFrame(rows)
        out_path = os.path.join(ds_dir, "comparison_summary.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n===== {ds.upper()} =====")
        print(out.to_string(index=False))
        print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    main()
