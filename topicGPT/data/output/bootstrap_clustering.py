import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICAS = ["harmonic_mean_purity_p1", "adjusted_rand_index",
            "normalized_mutual_info", "topic_alignment_pn"]
RODADA_CPS = "round_0_cps"
RODADAS_RANDOM = ["round_1_random", "round_2_random", "round_3_random"]
DATASETS = ["wiki", "bills"]
CONFIANCA = 0.95

SAIDA = os.path.join(BASE_DIR, "bootstrap_clustering_summary.csv")


def parse_list(val):
    if isinstance(val, list):
        return val
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def primeiro_rotulo(valor, vazio):
    itens = parse_list(valor)
    return itens[0].lower().strip() if itens else vazio


def carregar_rotulos(dataset):
    tabelas = {}
    for rodada in [RODADA_CPS] + RODADAS_RANDOM:
        caminho = os.path.join(BASE_DIR, dataset, rodada, "assignment_parsed.csv")
        df = pd.read_csv(caminho, encoding="utf-8-sig")
        df = df.drop_duplicates(subset="text")
        tabelas[rodada] = pd.DataFrame({
            "y_true": df["topics"].apply(primeiro_rotulo, vazio="_no_gt").to_numpy(),
            "y_pred": df["predicted_topics"].apply(primeiro_rotulo, vazio="_unassigned").to_numpy(),
        }, index=pd.Index(df["text"].to_numpy(), name="text"))

    comum = tabelas[RODADA_CPS].index
    for rodada in RODADAS_RANDOM:
        comum = comum.intersection(tabelas[rodada].index)

    y_true = tabelas[RODADA_CPS].loc[comum, "y_true"].to_numpy()
    preditos = {r: t.loc[comum, "y_pred"].to_numpy() for r, t in tabelas.items()}

    for rodada in RODADAS_RANDOM:
        outro = tabelas[rodada].loc[comum, "y_true"].to_numpy()
        if not np.array_equal(y_true, outro):
            sys.exit(f"{dataset}/{rodada}: ground truth diverge do CPS nos documentos pareados")
    if pd.isna(y_true).any() or any(pd.isna(v).any() for v in preditos.values()):
        sys.exit(f"{dataset}: rotulos ausentes apos o pareamento")

    return y_true, preditos


def codificar(y_true, y_pred):
    cod_true, _ = pd.factorize(y_true)
    cod_pred, rotulos_pred = pd.factorize(y_pred)
    n_pred = len(rotulos_pred)
    n_true = cod_true.max() + 1
    return cod_true * n_pred + cod_pred, n_true, n_pred


def metricas_contingencia(C):
    n = C.sum()
    a = C.sum(axis=1)
    b = C.sum(axis=0)

    soma_ij = (C * (C - 1)).sum() / 2.0
    soma_a = (a * (a - 1)).sum() / 2.0
    soma_b = (b * (b - 1)).sum() / 2.0
    esperado = soma_a * soma_b / (n * (n - 1) / 2.0)
    maximo = (soma_a + soma_b) / 2.0
    ari = 0.0 if maximo == esperado else (soma_ij - esperado) / (maximo - esperado)

    pa = a[a > 0] / n
    pb = b[b > 0] / n
    h_a = -(pa * np.log(pa)).sum()
    h_b = -(pb * np.log(pb)).sum()

    nz = C > 0
    if nz.any():
        c_nz = C[nz]
        externo = np.outer(a, b)[nz]
        mi = (c_nz / n * np.log(c_nz * n / externo)).sum()
    else:
        mi = 0.0
    nmi = 0.0 if h_a <= 0 or h_b <= 0 else mi / ((h_a + h_b) / 2.0)

    denominador = a[:, None] + b[None, :]
    f = np.divide(2.0 * C, denominador, out=np.zeros_like(C, dtype=float),
                  where=denominador > 0)
    melhor_f = f.max(axis=1) if f.shape[1] else np.zeros(len(a))
    p1 = float((a / n * melhor_f).sum())

    return np.array([p1, ari, nmi, (p1 + nmi) / 2.0])


def contingencia(codigo, contagens, n_true, n_pred):
    plano = np.bincount(codigo, weights=contagens, minlength=n_true * n_pred)
    return plano.reshape(n_true, n_pred)


def validar(y_true, y_pred, rotulo):
    codigo, n_true, n_pred = codificar(y_true, y_pred)
    C = contingencia(codigo, np.ones(len(y_true)), n_true, n_pred)
    p1, ari, nmi, pn = metricas_contingencia(C)

    ari_ref = adjusted_rand_score(y_true, y_pred)
    nmi_ref = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")

    for nome, obtido, esperado in [("ARI", ari, ari_ref), ("NMI", nmi, nmi_ref)]:
        if not np.isclose(obtido, esperado, atol=1e-9):
            sys.exit(f"{rotulo}: {nome} divergiu — {obtido:.10f} vs {esperado:.10f}")
    return p1, ari, nmi, pn


def ic_percentil(replicas):
    alfa = (1.0 - CONFIANCA) / 2.0
    return np.percentile(replicas, [100 * alfa, 100 * (1 - alfa)], axis=0)


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    ordem = np.argsort(p)
    escalado = p[ordem] * m / np.arange(1, m + 1)
    ajustado = np.empty(m)
    ajustado[ordem] = np.minimum.accumulate(escalado[::-1])[::-1]
    return np.clip(ajustado, 0, 1)


def processar(dataset, n_replicas, seed):
    y_true, preditos = carregar_rotulos(dataset)
    n = len(y_true)

    codificados = {}
    observado = {}
    for rodada, y_pred in preditos.items():
        codificados[rodada] = codificar(y_true, y_pred)
        observado[rodada] = validar(y_true, y_pred, f"{dataset}/{rodada}")

    obs_cps = np.array(observado[RODADA_CPS])
    obs_rnd = np.mean([observado[r] for r in RODADAS_RANDOM], axis=0)

    rng = np.random.default_rng(seed)
    pesos = np.full(n, 1.0 / n)
    rep_cps, rep_rnd = [], []

    for i in range(n_replicas):
        contagens = rng.multinomial(n, pesos).astype(float)
        valores = {}
        for rodada, (codigo, n_true, n_pred) in codificados.items():
            C = contingencia(codigo, contagens, n_true, n_pred)
            valores[rodada] = metricas_contingencia(C)
        rep_cps.append(valores[RODADA_CPS])
        rep_rnd.append(np.mean([valores[r] for r in RODADAS_RANDOM], axis=0))

        if (i + 1) % 2000 == 0:
            print(f"  {dataset}: {i + 1}/{n_replicas} replicas", flush=True)

    rep_cps = np.array(rep_cps)
    rep_rnd = np.array(rep_rnd)
    rep_delta = rep_cps - rep_rnd

    cps_inf, cps_sup = ic_percentil(rep_cps)
    rnd_inf, rnd_sup = ic_percentil(rep_rnd)
    d_inf, d_sup = ic_percentil(rep_delta)

    cauda_neg = (1 + (rep_delta <= 0).sum(axis=0)) / (n_replicas + 1)
    cauda_pos = (1 + (rep_delta >= 0).sum(axis=0)) / (n_replicas + 1)
    p_valor = np.minimum(1.0, 2 * np.minimum(cauda_neg, cauda_pos))

    linhas = []
    for i, metrica in enumerate(METRICAS):
        linhas.append({
            "dataset": dataset,
            "metrica": metrica,
            "n_pareado": n,
            "cps": obs_cps[i],
            "cps_ic_inf": cps_inf[i],
            "cps_ic_sup": cps_sup[i],
            "random_media": obs_rnd[i],
            "random_ic_inf": rnd_inf[i],
            "random_ic_sup": rnd_sup[i],
            "delta": obs_cps[i] - obs_rnd[i],
            "ic_inf": d_inf[i],
            "ic_sup": d_sup[i],
            "p_valor": p_valor[i],
        })
    return linhas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=10_000,
                        help="numero de replicas bootstrap (padrao: 10000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    linhas = []
    for dataset in DATASETS:
        linhas.extend(processar(dataset, args.replicas, args.seed))

    resumo = pd.DataFrame(linhas)
    resumo["p_bh"] = benjamini_hochberg(resumo["p_valor"])
    resumo["significativo"] = resumo["p_bh"] < 0.05
    resumo.to_csv(SAIDA, index=False, encoding="utf-8-sig")

    print(f"\nB = {args.replicas:,} replicas | IC de {CONFIANCA:.0%} | "
          f"correcao BH sobre {len(resumo)} testes")
    print(f"Salvo em: {SAIDA}\n")
    print(resumo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
