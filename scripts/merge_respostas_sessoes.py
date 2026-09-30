"""Junta as respostas do formulário (Google Forms) com os dados de sessão e os turnos
de conversa do Firestore, usando o código de sessão que o participante colou no formulário.

Uso:
    GOOGLE_APPLICATION_CREDENTIALS=/caminho/para/chave.json python3 scripts/merge_respostas_sessoes.py
"""

import os

import pandas as pd
from google.cloud import firestore
from openpyxl.utils import get_column_letter

CREDENTIALS_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/ericbfmariano/Downloads/ideias-em-rede-vhlab-firebase-adminsdk-fbsvc-d4b1531715.json",
)
RESPOSTAS_PATH = "/Users/ericbfmariano/Downloads/Copy of Memory and Empathy (respostas).xlsx"
CODIGO_COLUMN = "Cole o código que apareceu ao abrir o site:"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "deploy", "analise_respostas.xlsx")

# Códigos de sessão a descartar da análise (ex: não encontrados no Firestore, testes, etc).
CODIGOS_EXCLUIR = {"VH-3KDQ-K4ZB", "VH-GK4T-B079"}

# Descarta também respostas que chegaram sem nenhum código de sessão preenchido.
EXCLUIR_SEM_CODIGO = True

# Sessões a incluir mesmo sem resposta no formulário (ex: sessão nova ainda sem match no
# arquivo de respostas baixado). Entram com as perguntas do formulário em branco.
CODIGOS_ADICIONAR_SEM_RESPOSTA: set[str] = set()

# Correções manuais de grupo (quando a interação real do participante não bate com o
# valor gravado no Firestore). Confirmado: VH-JSKE-D5QX (Rodrigo) é masculino mesmo,
# igual ao Firestore — nenhuma correção necessária.
GRUPO_OVERRIDE: dict[str, str] = {}

# Respostas reconstruídas manualmente por terem chegado desalinhadas na cópia de trabalho,
# antes de constarem no export oficial do Forms. Ficam vazio agora que o export do Forms já
# traz VH-JSKE-D5QX corretamente — reative aqui se algum caso novo do tipo aparecer.
RESPOSTAS_MANUAIS: list[dict] = []


def flatten_turno(doc_id: str, sessao_id: str, data: dict) -> dict:
    reasoning = data.get("reasoning") or {}
    ferramentas = reasoning.get("ferramentas") or []
    nomes_ferramentas = ", ".join(f.get("nome", "") for f in ferramentas if isinstance(f, dict))
    return {
        "sessao_id": sessao_id,
        "ordem": data.get("ordem"),
        "query": data.get("query"),
        "resposta": data.get("resposta"),
        "decisao": reasoning.get("decisao"),
        "desfecho": reasoning.get("desfecho"),
        "ferramentas_usadas": nomes_ferramentas,
        "n_retrieves": len(data.get("retrieves") or []),
        "momento": data.get("momento"),
        "turno_id": doc_id,
    }


def carregar_sessoes_e_turnos(client: firestore.Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = list(client.collection("sessoes").stream())
    sessao_rows = []
    turno_rows = []
    for doc in docs:
        data = doc.to_dict() or {}
        turnos = list(doc.reference.collection("turnos").order_by("ordem").stream())
        sessao_rows.append(
            {
                "sessao_id": doc.id,
                "grupo": data.get("grupo"),
                "entrada": data.get("entrada"),
                "iniciado_em": data.get("iniciado_em"),
                "finalizado_em": data.get("finalizado_em"),
                "n_turnos": len(turnos),
            }
        )
        for turno in turnos:
            turno_rows.append(flatten_turno(turno.id, doc.id, turno.to_dict() or {}))
    return pd.DataFrame(sessao_rows), pd.DataFrame(turno_rows)


def autosize(ws, columns: list[str], widths: dict[str, int] | None = None, default: int = 18) -> None:
    widths = widths or {}
    for idx, col in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(col, default)
    ws.freeze_panes = "A2"


def main() -> None:
    client = firestore.Client.from_service_account_json(CREDENTIALS_PATH)

    respostas = pd.read_excel(RESPOSTAS_PATH)
    if RESPOSTAS_MANUAIS:
        respostas = pd.concat([respostas, pd.DataFrame(RESPOSTAS_MANUAIS)], ignore_index=True)
    respostas["sessao_id"] = respostas[CODIGO_COLUMN].astype(str).str.strip()
    respostas.loc[respostas["sessao_id"] == "nan", "sessao_id"] = None
    respostas = respostas[~respostas["sessao_id"].isin(CODIGOS_EXCLUIR)].reset_index(drop=True)
    duplicados = respostas["sessao_id"].dropna()
    duplicados = duplicados[duplicados.duplicated()].unique().tolist()
    if duplicados:
        print(f"AVISO: código(s) de sessão duplicado(s) nas respostas, mantendo a última: {duplicados}")
        respostas = respostas.drop_duplicates(subset="sessao_id", keep="last").reset_index(drop=True)
    if EXCLUIR_SEM_CODIGO:
        respostas = respostas[respostas["sessao_id"].notna()].reset_index(drop=True)

    sessoes, turnos = carregar_sessoes_e_turnos(client)

    merged = respostas.merge(sessoes, on="sessao_id", how="left", indicator=True)
    merged = merged.rename(columns={"_merge": "sessao_encontrada"})
    merged["sessao_encontrada"] = merged["sessao_encontrada"].map(
        {"both": "sim", "left_only": "nao", "right_only": "nao"}
    )

    for sessao_id, grupo_correto in GRUPO_OVERRIDE.items():
        merged.loc[merged["sessao_id"] == sessao_id, "grupo"] = grupo_correto

    # Coloca as colunas de junção logo no início, antes das perguntas do formulário.
    cols = list(merged.columns)
    front = ["sessao_id", "sessao_encontrada", "grupo", "entrada", "iniciado_em", "finalizado_em", "n_turnos"]
    ordered = front + [c for c in cols if c not in front]
    merged = merged[ordered]

    # Sessões adicionadas manualmente, sem resposta no formulário (perguntas ficam em branco).
    faltando = CODIGOS_ADICIONAR_SEM_RESPOSTA - set(merged["sessao_id"].dropna())
    if faltando:
        extras = sessoes[sessoes["sessao_id"].isin(faltando)].copy()
        extras["sessao_encontrada"] = "sim"
        merged = pd.concat([merged, extras], ignore_index=True)[ordered]

    # Só os turnos das sessões que responderam ao formulário.
    ids_respondentes = set(merged["sessao_id"].dropna())
    turnos_respondentes = turnos[turnos["sessao_id"].isin(ids_respondentes)].reset_index(drop=True)

    sem_codigo = respostas["sessao_id"].isna().sum()
    nao_encontrados = merged.loc[merged["sessao_encontrada"] == "nao", "sessao_id"].dropna().tolist()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="respostas_sessao")
        autosize(writer.sheets["respostas_sessao"], list(merged.columns))

        turnos_respondentes.to_excel(writer, index=False, sheet_name="turnos")
        autosize(
            writer.sheets["turnos"],
            list(turnos_respondentes.columns),
            widths={"sessao_id": 16, "query": 45, "resposta": 70, "momento": 22},
        )

    print(f"Juntadas {len(merged)} respostas com dados de sessão em {OUTPUT_PATH}")
    print(f"Turnos incluídos (apenas de sessões respondentes): {len(turnos_respondentes)}")
    print(f"Respostas sem código de sessão preenchido: {sem_codigo}")
    print(f"Códigos que não bateram com nenhuma sessão no Firestore: {nao_encontrados}")


if __name__ == "__main__":
    main()
