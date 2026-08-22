import argparse
import os

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer

from bootstrap_clustering import (BASE_DIR, CONFIANCA, DATASETS, RODADA_CPS,
                                  RODADAS_RANDOM, primeiro_rotulo)

SAIDA = os.path.join(BASE_DIR, "bootstrap_coerencia_summary.csv")
TOP_N = 10


def carregar_textos(dataset):
    tabelas = {}
    for rodada in [RODADA_CPS] + RODADAS_RANDOM:
        caminho = os.path.join(BASE_DIR, dataset, rodada, "assignment_parsed.csv")
        df = pd.read_csv(caminho, encoding="utf-8-sig").drop_duplicates(subset="text")
        previsto = df["predicted_topics"].apply(primeiro_rotulo, vazio="_unassigned")
        tabelas[rodada] = pd.Series(previsto.to_numpy(),
                                    index=pd.Index(df["text"].astype(str).to_numpy()))

    comum = tabelas[RODADA_CPS].index
    for rodada in RODADAS_RANDOM:
        comum = comum.intersection(tabelas[rodada].index)

    textos = list(comum)
    return textos, {r: t.loc[comum].to_numpy() for r, t in tabelas.items()}


def palavras_por_topico(textos, previstos):
    por_topico = {}
    for texto, topico in zip(textos, previstos):
        if topico != "_unassigned":
            por_topico.setdefault(topico, []).append(texto)

    topicos_palavras = []
    for docs in por_topico.values():
        if len(docs) < 2:
            continue
        try:
            vetor = TfidfVectorizer(max_features=TOP_N, stop_words="english",
                                    token_pattern=r"[a-z]{2,}")
            vetor.fit(docs)
            palavras = vetor.get_feature_names_out().tolist()
        except ValueError:
            continue
        if len(palavras) >= 5:
            topicos_palavras.append(palavras)
    return topicos_palavras


def coerencia_por_topico(topicos_palavras, tokenizados, dicionario):
    if not topicos_palavras:
        return np.array([])
    modelo = CoherenceModel(topics=topicos_palavras, texts=tokenizados,
                            dictionary=dicionario, coherence="c_v")
    return np.array(modelo.get_coherence_per_topic(), dtype=float)


def bootstrap_media(valores, n_replicas, rng):
    if valores.size == 0:
        return np.zeros(n_replicas)
    indices = rng.integers(0, valores.size, size=(n_replicas, valores.size))
    return valores[indices].mean(axis=1)


def ic_percentil(replicas):
    alfa = (1.0 - CONFIANCA) / 2.0
    return np.percentile(replicas, [100 * alfa, 100 * (1 - alfa)])


def processar(dataset, n_replicas, seed):
    textos, previstos = carregar_textos(dataset)
    print(f"\n>>> {dataset}: {len(textos):,} documentos pareados", flush=True)

    tokenizados = [t.lower().split() for t in textos]
    dicionario = Dictionary(tokenizados)

    coerencias = {}
    for rodada in [RODADA_CPS] + RODADAS_RANDOM:
        topicos_palavras = palavras_por_topico(textos, previstos[rodada])
        valores = coerencia_por_topico(topicos_palavras, tokenizados, dicionario)
        coerencias[rodada] = valores
        print(f"  {rodada}: {len(valores)} topicos | C_v = {valores.mean():.4f}", flush=True)

    rng = np.random.default_rng(seed)
    rep_cps = bootstrap_media(coerencias[RODADA_CPS], n_replicas, rng)
    rep_rnd = np.mean([bootstrap_media(coerencias[r], n_replicas, rng)
                       for r in RODADAS_RANDOM], axis=0)
    rep_delta = rep_cps - rep_rnd

    obs_cps = float(coerencias[RODADA_CPS].mean())
    obs_rnd = float(np.mean([coerencias[r].mean() for r in RODADAS_RANDOM]))

    cps_inf, cps_sup = ic_percentil(rep_cps)
    rnd_inf, rnd_sup = ic_percentil(rep_rnd)
    d_inf, d_sup = ic_percentil(rep_delta)

    cauda_neg = (1 + (rep_delta <= 0).sum()) / (n_replicas + 1)
    cauda_pos = (1 + (rep_delta >= 0).sum()) / (n_replicas + 1)

    return {
        "dataset": dataset,
        "metrica": "topic_coherence_cv",
        "n_topicos_cps": int(coerencias[RODADA_CPS].size),
        "n_topicos_random_medio": float(np.mean([coerencias[r].size for r in RODADAS_RANDOM])),
        "cps": obs_cps,
        "cps_ic_inf": cps_inf,
        "cps_ic_sup": cps_sup,
        "random_media": obs_rnd,
        "random_ic_inf": rnd_inf,
        "random_ic_sup": rnd_sup,
        "delta": obs_cps - obs_rnd,
        "ic_inf": d_inf,
        "ic_sup": d_sup,
        "p_valor": float(min(1.0, 2 * min(cauda_neg, cauda_pos))),
        "reamostragem": "topicos (nao pareado)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    linhas = [processar(ds, args.replicas, args.seed) for ds in DATASETS]

    resumo = pd.DataFrame(linhas)
    resumo["significativo"] = resumo["p_valor"] < 0.05
    resumo.to_csv(SAIDA, index=False, encoding="utf-8-sig")

    print(f"\nB = {args.replicas:,} replicas | IC de {CONFIANCA:.0%}")
    print(f"Salvo em: {SAIDA}\n")
    print(resumo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
