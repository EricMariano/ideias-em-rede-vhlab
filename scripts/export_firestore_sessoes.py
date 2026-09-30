"""Exporta a coleção "sessoes" (e a subcoleção "turnos" de cada sessão) do Firestore
para uma planilha .xlsx com duas abas.

Uso:
    GOOGLE_APPLICATION_CREDENTIALS=/caminho/para/chave.json python3 scripts/export_firestore_sessoes.py

Se a variável de ambiente não for definida, usa CREDENTIALS_PATH abaixo como padrão.
"""

import os

import pandas as pd
from google.cloud import firestore
from openpyxl.utils import get_column_letter

CREDENTIALS_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/ericbfmariano/Downloads/ideias-em-rede-vhlab-firebase-adminsdk-fbsvc-d4b1531715.json",
)
COLLECTION_NAME = "sessoes"
SUBCOLLECTION_NAME = "turnos"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "deploy", "sessoes_export.xlsx")

# Larguras de coluna pensadas para leitura do conteúdo das conversas (query/resposta).
COLUMN_WIDTHS = {
    "sessoes": {"id": 16, "grupo": 10, "entrada": 10, "url": 45, "iniciado_em": 22, "finalizado_em": 22},
    "turnos": {
        "sessao_id": 16,
        "ordem": 6,
        "grupo": 10,
        "query": 45,
        "resposta": 70,
        "decisao": 16,
        "desfecho": 14,
        "ferramentas_usadas": 20,
        "n_retrieves": 12,
        "momento": 22,
    },
}


def flatten_sessao(doc_id: str, data: dict) -> dict:
    row = {"id": doc_id}
    for key, value in data.items():
        row[key] = str(value) if isinstance(value, (dict, list)) else value
    return row


def flatten_turno(doc_id: str, sessao_id: str, data: dict) -> dict:
    reasoning = data.get("reasoning") or {}
    ferramentas = reasoning.get("ferramentas") or []
    nomes_ferramentas = ", ".join(f.get("nome", "") for f in ferramentas if isinstance(f, dict))
    return {
        "sessao_id": sessao_id,
        "ordem": data.get("ordem"),
        "grupo": data.get("grupo"),
        "query": data.get("query"),
        "resposta": data.get("resposta"),
        "decisao": reasoning.get("decisao"),
        "desfecho": reasoning.get("desfecho"),
        "ferramentas_usadas": nomes_ferramentas,
        "n_retrieves": len(data.get("retrieves") or []),
        "momento": data.get("momento"),
        "turno_id": doc_id,
    }


def autosize(writer: pd.ExcelWriter, sheet_name: str, columns: list[str]) -> None:
    widths = COLUMN_WIDTHS.get(sheet_name, {})
    ws = writer.sheets[sheet_name]
    for idx, col in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(col, 18)
    ws.freeze_panes = "A2"


def main() -> None:
    client = firestore.Client.from_service_account_json(CREDENTIALS_PATH)
    session_docs = list(client.collection(COLLECTION_NAME).stream())

    if not session_docs:
        print(f"Nenhum documento encontrado na coleção '{COLLECTION_NAME}'.")
        return

    session_rows = [flatten_sessao(doc.id, doc.to_dict() or {}) for doc in session_docs]

    turno_rows = []
    for doc in session_docs:
        turnos = doc.reference.collection(SUBCOLLECTION_NAME).order_by("ordem").stream()
        for turno in turnos:
            turno_rows.append(flatten_turno(turno.id, doc.id, turno.to_dict() or {}))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df_sessoes = pd.DataFrame(session_rows)
        df_sessoes.to_excel(writer, index=False, sheet_name=COLLECTION_NAME)
        autosize(writer, COLLECTION_NAME, list(df_sessoes.columns))

        if turno_rows:
            df_turnos = pd.DataFrame(turno_rows)
            df_turnos.to_excel(writer, index=False, sheet_name=SUBCOLLECTION_NAME)
            autosize(writer, SUBCOLLECTION_NAME, list(df_turnos.columns))

    print(f"Exportadas {len(session_rows)} sessões e {len(turno_rows)} turnos para {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
