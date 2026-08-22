import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICAS = ["exact_match_pct", "cosine_similarity", "bert_score_f1",
            "jaccard_similarity", "levenshtein_similarity"]
RODADA_CPS = "round_0_cps"
RODADAS_RANDOM = ["round_1_random", "round_2_random", "round_3_random"]
DATASETS = ["wiki", "bills"]
B_REPLICAS = 10_000
SEED_BOOT = 42
CONFIANCA = 0.95

SUMMARY_PATH = os.path.join(BASE_DIR, "bootstrap_paired_summary.csv")


def carregar_pareado(dataset):
    tabelas = {}
    for rodada in [RODADA_CPS] + RODADAS_RANDOM:
        caminho = os.path.join(BASE_DIR, dataset, rodada, "assignment_parsed.csv")
        df = pd.read_csv(caminho, encoding="utf-8-sig")
        tabelas[rodada] = df.drop_duplicates(subset="text").set_index("text")[METRICAS]

    comum = tabelas[RODADA_CPS].index
    for rodada in RODADAS_RANDOM:
        comum = comum.intersection(tabelas[rodada].index)

    return {r: t.loc[comum].to_numpy(dtype=float) for r, t in tabelas.items()}


def bootstrap_media(X, n_replicas=B_REPLICAS, seed=SEED_BOOT, chunk=200):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    pesos = np.full(n, 1.0 / n)
    replicas = []
    for inicio in range(0, n_replicas, chunk):
        contagens = rng.multinomial(n, pesos, size=min(chunk, n_replicas - inicio))
        replicas.append(contagens.astype(float) @ X / n)
    return np.vstack(replicas)


def ic_percentil(replicas):
    alfa = (1.0 - CONFIANCA) / 2.0
    return np.percentile(replicas, [100 * alfa, 100 * (1 - alfa)], axis=0)


def ic_t_student(valores, rotulo):
    n_runs = valores.shape[0]
    if n_runs < 2:
        raise ValueError(f"{rotulo}: {n_runs} execucao(oes), IC exige ao menos 2")

    media = valores.mean(axis=0)
    erro_padrao = valores.std(axis=0, ddof=1) / np.sqrt(n_runs)
    t_critico = stats.t.ppf(1 - (1 - CONFIANCA) / 2, df=n_runs - 1)

    margem = t_critico * erro_padrao
    return media, media - margem, media + margem, n_runs


def valores_bertopic(dataset):
    caminho = os.path.join(BASE_DIR, dataset, "bertopic_metrics.csv")
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    execucoes = df[df["run"].str.startswith("run_")]
    return execucoes[METRICAS].to_numpy(dtype=float), caminho


def main():
    if not os.path.exists(SUMMARY_PATH):
        sys.exit(f"Nao encontrado: {SUMMARY_PATH}\nRode antes a celula Bootstrap do notebook.")

    resumo = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    novas = {}

    for dataset in DATASETS:
        dados = carregar_pareado(dataset)
        media_random = np.mean([dados[r] for r in RODADAS_RANDOM], axis=0)

        cps_inf, cps_sup = ic_percentil(bootstrap_media(dados[RODADA_CPS]))
        rnd_inf, rnd_sup = ic_percentil(bootstrap_media(media_random))

        por_rodada = np.array([dados[r].mean(axis=0) for r in RODADAS_RANDOM])
        _, rnd_t_inf, rnd_t_sup, n_rnd = ic_t_student(por_rodada, f"{dataset}/random")

        bt_valores, bt_caminho = valores_bertopic(dataset)
        bt_media, bt_inf, bt_sup, n_bt = ic_t_student(bt_valores, bt_caminho)

        for i, metrica in enumerate(METRICAS):
            novas[(dataset, metrica)] = {
                "cps_ic_inf": cps_inf[i],
                "cps_ic_sup": cps_sup[i],
                "random_ic_inf": rnd_inf[i],
                "random_ic_sup": rnd_sup[i],
                "random_t_ic_inf": rnd_t_inf[i],
                "random_t_ic_sup": rnd_t_sup[i],
                "random_n_runs": n_rnd,
                "bertopic": bt_media[i],
                "bertopic_ic_inf": bt_inf[i],
                "bertopic_ic_sup": bt_sup[i],
                "bertopic_n_runs": n_bt,
            }

        print(f"{dataset}: {dados[RODADA_CPS].shape[0]:,} documentos pareados | "
              f"random com {n_rnd} execucoes | BERTopic com {n_bt} execucoes")

    chaves = list(zip(resumo["dataset"], resumo["metrica"]))
    faltantes = [c for c in chaves if c not in novas]
    if faltantes:
        sys.exit(f"Sem IC calculado para: {faltantes}")

    resumo = resumo.drop(columns=["ic_tipo_cps_random", "ic_tipo_bertopic"], errors="ignore")

    for coluna in ["cps_ic_inf", "cps_ic_sup",
                   "random_ic_inf", "random_ic_sup",
                   "random_t_ic_inf", "random_t_ic_sup", "random_n_runs",
                   "bertopic", "bertopic_ic_inf", "bertopic_ic_sup", "bertopic_n_runs"]:
        resumo[coluna] = [novas[c][coluna] for c in chaves]

    resumo["ic_tipo_bootstrap"] = "bootstrap sobre documentos"
    resumo["ic_tipo_t"] = "t-Student sobre execucoes"

    resumo.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(f"\nB = {B_REPLICAS:,} replicas | IC de {CONFIANCA:.0%}")
    print(f"Atualizado: {SUMMARY_PATH}\n")

    print("--- bootstrap sobre documentos ---")
    print(resumo[["dataset", "metrica", "cps", "cps_ic_inf", "cps_ic_sup",
                  "random_media", "random_ic_inf", "random_ic_sup"]]
          .round(4).to_string(index=False))

    print("\n--- t-Student sobre execucoes ---")
    print(resumo[["dataset", "metrica",
                  "random_media", "random_t_ic_inf", "random_t_ic_sup",
                  "bertopic", "bertopic_ic_inf", "bertopic_ic_sup"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
