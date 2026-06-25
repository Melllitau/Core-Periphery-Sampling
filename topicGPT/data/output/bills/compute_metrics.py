import sys
import ast
import os

sys.stdout.reconfigure(line_buffering=True)
import itertools
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score_fn

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


def compute_exact_match(topics, predicted):
    if not topics:
        return 0.0
    if not predicted:
        return 0.0
    gt_set = {t.lower().strip() for t in topics}
    pred_set = {t.lower().strip() for t in predicted}
    matches = len(gt_set & pred_set)
    return (matches / len(gt_set)) * 100.0


def compute_cosine_combinatorial(topics, predicted, embeddings_cache, model):
    if not topics or not predicted:
        return 0.0

    all_strings = list(set(topics + predicted))
    missing = [s for s in all_strings if s not in embeddings_cache]
    if missing:
        embs = model.encode(missing, convert_to_numpy=True)
        for s, emb in zip(missing, embs):
            embeddings_cache[s] = emb

    sims = []
    for t, p in itertools.product(topics, predicted):
        e1 = embeddings_cache[t]
        e2 = embeddings_cache[p]
        cos = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
        sims.append(float(cos))

    return sum(sims) / len(sims)


def jaccard_sim(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def levenshtein_sim(a, b):
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[m] / max_len


def compute_jaccard_combinatorial(topics, predicted):
    if not topics or not predicted:
        return 0.0
    sims = [jaccard_sim(t, p) for t, p in itertools.product(topics, predicted)]
    return sum(sims) / len(sims)


def compute_levenshtein_combinatorial(topics, predicted):
    if not topics or not predicted:
        return 0.0
    sims = [levenshtein_sim(t, p) for t, p in itertools.product(topics, predicted)]
    return sum(sims) / len(sims)


def compute_bert_scores_batch(all_topics, all_predicted):

    refs = []
    cands = []
    row_indices = []
    pair_counts = []

    for i, (topics, predicted) in enumerate(zip(all_topics, all_predicted)):
        if not topics or not predicted:
            pair_counts.append(0)
            continue
        pairs = list(itertools.product(topics, predicted))
        pair_counts.append(len(pairs))
        for t, p in pairs:
            refs.append(t)
            cands.append(p)
            row_indices.append(i)

    n_rows = len(all_topics)
    results = [0.0] * n_rows

    if not refs:
        return results

    valid = [(c, r, ri) for c, r, ri in zip(cands, refs, row_indices) if c.strip() and r.strip()]
    if not valid:
        return results
    cands_clean = [v[0] for v in valid]
    refs_clean = [v[1] for v in valid]
    row_indices_clean = [v[2] for v in valid]

    print(f"    Calculando BERTScore para {len(cands_clean)} pares...")
    _, _, f1 = bert_score_fn(cands_clean, refs_clean, lang="en", verbose=False, batch_size=256)
    f1 = f1.numpy()

    from collections import defaultdict
    row_scores = defaultdict(list)
    for score_val, ri in zip(f1, row_indices_clean):
        row_scores[ri].append(float(score_val))
    for ri, scores in row_scores.items():
        results[ri] = sum(scores) / len(scores)

    return results


def process_round(round_name, model):
    csv_path = os.path.join(BASE_DIR, round_name, "assignment_parsed.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return None

    print(f"\n>>> {round_name}")
    df = pd.read_csv(csv_path)

    df["_topics"] = df["topics"].apply(parse_list)
    df["_predicted"] = df["predicted_topics"].apply(parse_list)


    print("  Calculando exact_match_pct...")
    df["exact_match_pct"] = df.apply(
        lambda r: compute_exact_match(r["_topics"], r["_predicted"]), axis=1
    )

    print("  Calculando cosine_similarity...")
    embeddings_cache = {}
    cosines = []
    for _, row in df.iterrows():
        cosines.append(
            compute_cosine_combinatorial(row["_topics"], row["_predicted"], embeddings_cache, model)
        )
    df["cosine_similarity"] = cosines

    print("  Calculando bert_score_f1...")
    df["bert_score_f1"] = compute_bert_scores_batch(
        df["_topics"].tolist(), df["_predicted"].tolist()
    )

    print("  Calculando jaccard_similarity...")
    df["jaccard_similarity"] = df.apply(
        lambda r: compute_jaccard_combinatorial(r["_topics"], r["_predicted"]), axis=1
    )

    print("  Calculando levenshtein_similarity...")
    df["levenshtein_similarity"] = df.apply(
        lambda r: compute_levenshtein_combinatorial(r["_topics"], r["_predicted"]), axis=1
    )

    df.drop(columns=["_topics", "_predicted"], inplace=True, errors="ignore")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"  exact_match_pct        — mean: {df['exact_match_pct'].mean():.1f}%")
    print(f"  cosine_similarity      — mean: {df['cosine_similarity'].mean():.3f}")
    print(f"  bert_score_f1          — mean: {df['bert_score_f1'].mean():.3f}")
    print(f"  jaccard_similarity     — mean: {df['jaccard_similarity'].mean():.3f}")
    print(f"  levenshtein_similarity — mean: {df['levenshtein_similarity'].mean():.3f}")
    print(f"  Salvo em: {csv_path}")


def main():
    print("Carregando SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for round_name in ROUNDS:
        process_round(round_name, model)

    print("\nConcluído!")


if __name__ == "__main__":
    main()
