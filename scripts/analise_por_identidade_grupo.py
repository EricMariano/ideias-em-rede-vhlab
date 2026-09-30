"""Média e desvio padrão por combinação (identidade do participante x personagem/grupo
com quem interagiu), questão por questão e por subescala do Godspeed.
"""

import pandas as pd

INPUT_PATH = "/Users/ericbfmariano/Projects/ideias/deploy/analise_respostas.xlsx"
OUTPUT_PATH = "/Users/ericbfmariano/Projects/ideias/deploy/analise_identidade_grupo.xlsx"

COL_IDENTIDADE = "Quais das seguintes alternativas descreve a forma como você se identifica hoje?"

df = pd.read_excel(INPUT_PATH, sheet_name="respostas_sessao")
qcols = [c for c in df.columns if c.strip("ㅤ ") == "" or c.startswith("ㅤ")]

SUBESCALAS = {
    "Antropomorfismo": qcols[0:5],
    "Expressão de vida": qcols[5:11],
    "Simpatia": qcols[11:16],
    "Inteligência percebida": qcols[16:21],
    "Segurança percebida": qcols[21:24],
}
for nome, cols in SUBESCALAS.items():
    df[nome] = df[cols].mean(axis=1, skipna=True)

df["combo"] = df[COL_IDENTIDADE].astype(str) + " x personagem " + df["grupo"].astype(str)
COMBOS = [
    "Mulher Cis x personagem feminino",
    "Mulher Cis x personagem masculino",
    "Homem Cis x personagem feminino",
    "Homem Cis x personagem masculino",
]


def resumo(itens: dict[str, str], label_col: str) -> pd.DataFrame:
    rows = []
    for label, col in itens.items():
        for combo in COMBOS:
            vals = df.loc[df["combo"] == combo, col].dropna()
            rows.append(
                {
                    label_col: label,
                    "combo": combo,
                    "n": len(vals),
                    "media": round(vals.mean(), 2) if len(vals) else None,
                    "desvio_padrao": round(vals.std(ddof=1), 2) if len(vals) > 1 else None,
                }
            )
    return pd.DataFrame(rows)


por_questao = resumo({f"Q{i}": col for i, col in enumerate(qcols, start=1)}, "questao")
por_subescala = resumo({nome: nome for nome in SUBESCALAS}, "subescala")

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    por_questao.to_excel(writer, index=False, sheet_name="por_questao")
    por_subescala.to_excel(writer, index=False, sheet_name="por_subescala")

pd.set_option("display.width", 160)
print("--- Por subescala ---")
print(por_subescala.to_string(index=False))
print(f"\nSalvo em {OUTPUT_PATH}")
