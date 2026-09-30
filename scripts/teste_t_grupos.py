"""Teste t (statsmodels) comparando feminino x masculino, questão por questão."""

import pandas as pd
from statsmodels.stats.weightstats import ttest_ind

INPUT_PATH = "/Users/ericbfmariano/Projects/ideias/deploy/analise_respostas.xlsx"

df = pd.read_excel(INPUT_PATH, sheet_name="respostas_sessao")
cols = [c for c in df.columns if c.strip("ㅤ ") == "" or c.startswith("ㅤ")]

fem = df[df["grupo"] == "feminino"]
masc = df[df["grupo"] == "masculino"]

rows = []
for i, col in enumerate(cols, start=1):
    a = fem[col].dropna()
    b = masc[col].dropna()
    tstat, pvalue, df_ = ttest_ind(a, b, usevar="unequal")
    rows.append(
        {
            "questao": f"Q{i}",
            "media_feminino": round(a.mean(), 2),
            "media_masculino": round(b.mean(), 2),
            "n_feminino": len(a),
            "n_masculino": len(b),
            "t_stat": round(tstat, 3),
            "p_valor": round(pvalue, 4),
            "significativo_5pct": "sim" if pvalue < 0.05 else "nao",
        }
    )

resultado = pd.DataFrame(rows)

pd.set_option("display.width", 160)
print(resultado.to_string(index=False))

OUTPUT_PATH = "/Users/ericbfmariano/Projects/ideias/deploy/teste_t_feminino_x_masculino.xlsx"
resultado.to_excel(OUTPUT_PATH, index=False)
print(f"\nSalvo em {OUTPUT_PATH}")
