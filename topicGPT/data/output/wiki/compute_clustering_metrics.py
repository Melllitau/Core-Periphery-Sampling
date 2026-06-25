import sys
sys.stdout.reconfigure(line_buffering=True)

import ast
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROUNDS = [
    "round_0_cps",
    "round_1_random",
    "round_2_random",
    "round_3_random",
]


def parse_list(val):
    if isinstance(val, list):
        return val
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def harmonic_mean_purity(y_true, y_pred):
    N = len(y_true)
    classes = set(y_true)
    clusters = set(y_pred)
    contingency = Counter(zip(y_true, y_pred))

    cluster_sizes = Counter(y_pred)

    p1 = 0.0
    for c in classes:
        n_c = sum(1 for y in y_true if y == c)
        best_f = 0.0
        for w in clusters:
            n_cw = contingency.get((c, w), 0)
            if n_cw == 0:
                continue
            precision = n_cw / cluster_sizes[w]
            recall = n_cw / n_c
            f = 2 * precision * recall / (precision + recall)
            if f > best_f:
                best_f = f
        p1 += (n_c / N) * best_f

    return p1


def compute_purity(y_true, y_pred):
    N = len(y_true)
    contingency = Counter(zip(y_pred, y_true))
    clusters = set(y_pred)
    true_classes = set(y_true)
    purity = sum(
        max(contingency.get((w, c), 0) for c in true_classes)
        for w in clusters
    ) / N
    return purity


def compute_cv(texts, predicted_labels, top_n=10):
    tokenized_docs = [doc.lower().split() for doc in texts]
    dictionary = Dictionary(tokenized_docs)

    unique_topics = set(predicted_labels)
    unique_topics.discard("_unassigned")
    topics_words = []

    for topic in unique_topics:
        topic_docs = [texts[i] for i in range(len(texts)) if predicted_labels[i] == topic]
        if len(topic_docs) < 2:
            continue
        try:
            vectorizer = TfidfVectorizer(max_features=top_n, stop_words="english",
                                         token_pattern=r"[a-z]{2,}")
            tfidf = vectorizer.fit_transform(topic_docs)
            words = vectorizer.get_feature_names_out().tolist()
            if len(words) >= 5:
                topics_words.append(words)
        except ValueError:
            continue

    if not topics_words:
        return 0.0

    cm = CoherenceModel(
        topics=topics_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    return cm.get_coherence()


def process_round(round_name):
    csv_path = os.path.join(BASE_DIR, round_name, "assignment_parsed.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return None

    print(f"\n>>> {round_name}")
    df = pd.read_csv(csv_path)

    topics_parsed = df["topics"].apply(parse_list)
    predicted_parsed = df["predicted_topics"].apply(parse_list)

    y_true = []
    y_pred = []
    texts = []
    for i in range(len(df)):
        gt = topics_parsed.iloc[i]
        pr = predicted_parsed.iloc[i]
        gt_label = gt[0].lower().strip() if gt else "_no_gt"
        pr_label = pr[0].lower().strip() if pr else "_unassigned"
        y_true.append(gt_label)
        y_pred.append(pr_label)
        texts.append(str(df["text"].iloc[i]))

    print(f"  Documentos: {len(y_true)} | sem GT: {y_true.count('_no_gt')} | sem pred: {y_pred.count('_unassigned')}")

    print("  Calculando P1...")
    p1 = harmonic_mean_purity(y_true, y_pred)

    print("  Calculando ARI...")
    ari = adjusted_rand_score(y_true, y_pred)

    print("  Calculando NMI...")
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")

    print("  Calculando C_V...")
    cv = compute_cv(texts, y_pred)

    print("  Calculando Purity e PN...")
    purity = compute_purity(y_true, y_pred)
    pn = (p1 + nmi) / 2

    print(f"  harmonic_mean_purity_p1  — {p1:.4f}")
    print(f"  adjusted_rand_index      — {ari:.4f}")
    print(f"  normalized_mutual_info   — {nmi:.4f}")
    print(f"  topic_coherence_cv       — {cv:.4f}")
    print(f"  topic_alignment_pn       — {pn:.4f}")
    print(f"  (purity={purity:.4f})")

    return {
        "round": round_name,
        "harmonic_mean_purity_p1": p1,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "topic_coherence_cv": cv,
        "topic_alignment_pn": pn,
    }


def main():
    rows = []
    for round_name in ROUNDS:
        result = process_round(round_name)
        if result:
            rows.append(result)

    if not rows:
        print("Nenhum round processado.")
        return

    clust_df = pd.DataFrame(rows)
    clust_path = os.path.join(BASE_DIR, "clustering_metrics.csv")
    clust_df.to_csv(clust_path, index=False, encoding="utf-8-sig")
    print(f"\nSalvo em: {clust_path}")
    print(clust_df.to_string(index=False))


if __name__ == "__main__":
    main()
