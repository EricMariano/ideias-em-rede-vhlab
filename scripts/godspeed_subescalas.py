"""Médias/desvios padrão das 5 subescalas do Godspeed Questionnaire (Geral, Feminino,
Masculino) e teste t de Welch (feminino x masculino) por subescala.
"""

import pandas as pd
from statsmodels.stats.weightstats import ttest_ind

INPUT_PATH = "/Users/ericbfmariano/Projects/ideias/deploy/analise_respostas.xlsx"
OUTPUT_PATH = "/Users/ericbfmariano/Projects/ideias/deploy/godspeed_subescalas.xlsx"

df = pd.read_excel(INPUT_PATH, sheet_name="respostas_sessao")
qcols = [c for c in df.columns if c.strip("ㅤ ") == "" or c.startswith("ㅤ")]

# Ordem padrão do Godspeed Questionnaire, que bate com os 24 itens do formulário.
SUBESCALAS = {
    "Antropomorfismo": qcols[0:5],
    "Expressão de vida": qcols[5:11],
    "Simpatia": qcols[11:16],
    "Inteligência percebida": qcols[16:21],
    "Segurança percebida": qcols[21:24],
}

for nome, cols in SUBESCALAS.items():
    df[nome] = df[cols].mean(axis=1, skipna=True)

fem = df[df["grupo"] == "feminino"]
masc = df[df["grupo"] == "masculino"]

rows = []
for nome in SUBESCALAS:
    a = fem[nome].dropna()
    b = masc[nome].dropna()
    tstat, pvalue, _ = ttest_ind(a, b, usevar="unequal")  # Welch, sem correção de múltiplas comparações
    rows.append(
        {
            "subescala": nome,
            "n_geral": df[nome].notna().sum(),
            "media_geral": round(df[nome].mean(), 2),
            "dp_geral": round(df[nome].std(ddof=1), 2),
            "n_feminino": len(a),
            "media_feminino": round(a.mean(), 2),
            "dp_feminino": round(a.std(ddof=1), 2),
            "n_masculino": len(b),
            "media_masculino": round(b.mean(), 2),
            "dp_masculino": round(b.std(ddof=1), 2),
            "t_stat": round(tstat, 3),
            "p_valor": round(pvalue, 4),
            "significativo_5pct": "sim" if pvalue < 0.05 else "nao",
        }
    )

resultado = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(resultado.to_string(index=False))

resultado.to_excel(OUTPUT_PATH, index=False)
print(f"\nSalvo em {OUTPUT_PATH}")
