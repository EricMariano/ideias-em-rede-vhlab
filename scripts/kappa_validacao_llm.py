"""Concordância entre a anotação da LLM e a de dois anotadores humanos, campo a campo.

Lê deploy/analise_llm/validacao_humana.xlsx (gerada por analise_llm_interacoes.py) depois que
os anotadores preencheram as abas "anotador_1" e "anotador_2". Reporta o kappa de Cohen para
cada par (humano x humano, LLM x cada humano) e, nas escalas ordinais, o kappa ponderado
quadrático, que não pune igual errar por um ponto e errar por quatro.

Uso:
    python3 scripts/kappa_validacao_llm.py
"""

import os

import numpy as np
import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "deploy", "analise_llm", "validacao_humana.xlsx")
ORDINAIS = {"sentimento_usuario", "abuso_severidade"}
CAMPOS = ["sentimento_usuario", "abuso_severidade", "intencao", "concepcao_agente",
          "estrategia_realinhamento", "resposta_atende"]


def cohen_kappa_score(x: pd.Series, y: pd.Series, weights: str | None = None) -> float:
    """Kappa de Cohen; com weights="quadratic", a versão ponderada para escalas ordinais."""
    categorias = sorted(set(x) | set(y))
    idx = {c: i for i, c in enumerate(categorias)}
    k = len(categorias)
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[idx[a], idx[b]] += 1
    obs /= obs.sum()
    esp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weights == "quadratic":
        pos = np.array(categorias, dtype=float)
        w = (pos[:, None] - pos[None, :]) ** 2
    else:
        w = 1 - np.eye(k)
    den = (w * esp).sum()
    return 1.0 if den == 0 else 1 - (w * obs).sum() / den


def normalizar(s: pd.Series, ordinal: bool) -> pd.Series:
    if ordinal:
        return pd.to_numeric(s, errors="coerce")
    return s.astype("string").str.strip().str.lower()


def main() -> None:
    abas = {nome: pd.read_excel(PATH, sheet_name=nome).set_index("turno_id")
            for nome in ("anotador_1", "anotador_2", "llm")}
    pares = [("anotador_1", "anotador_2"), ("llm", "anotador_1"), ("llm", "anotador_2")]

    linhas = []
    for campo in CAMPOS:
        ordinal = campo in ORDINAIS
        for a, b in pares:
            x = normalizar(abas[a][campo], ordinal)
            y = normalizar(abas[b][campo], ordinal).reindex(x.index)
            ok = x.notna() & y.notna()
            if ok.sum() < 2:
                linhas.append({"campo": campo, "par": f"{a} x {b}", "n": int(ok.sum())})
                continue
            linha = {
                "campo": campo,
                "par": f"{a} x {b}",
                "n": int(ok.sum()),
                "concordancia_bruta": round((x[ok] == y[ok]).mean(), 3),
                "kappa": round(cohen_kappa_score(x[ok].astype(str), y[ok].astype(str)), 3),
            }
            if ordinal:
                linha["kappa_ponderado"] = round(cohen_kappa_score(x[ok], y[ok], weights="quadratic"), 3)
            linhas.append(linha)

    df = pd.DataFrame(linhas)
    print(df.to_string(index=False))
    saida = PATH.replace(".xlsx", "_kappa.xlsx")
    df.to_excel(saida, index=False)
    print(f"\nSalvo em {saida}")
    print("Referência (Landis & Koch): 0,41–0,60 moderada · 0,61–0,80 substancial · > 0,80 quase perfeita.")


if __name__ == "__main__":
    main()
