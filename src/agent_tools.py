"""
Módulo de ferramentas (@tool) do LangChain para o RAG Agêntico de Auditoria Parlamentar.

Este módulo disponibiliza 3 ferramentas especializadas para busca vetorial no ChromaDB
com suporte a filtragem dinâmica por metadados e formatação estruturada dos retornos.
"""

import os
import logging
from functools import lru_cache
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

from src.embeddings import get_embedding_model
from src.vector_store import get_chroma_client

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_vector_store(collection_name: str) -> Chroma:
    """Auxiliar para conectar à coleção do ChromaDB resolvendo o caminho persistido."""
    possible_paths = [
        "notebooks/data/processed/chroma_db",
        "data/processed/chroma_db",
        "../data/processed/chroma_db",
    ]
    db_path = "notebooks/data/processed/chroma_db"
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "chroma.sqlite3")):
            db_path = p
            break

    client = get_chroma_client(db_path)
    embedding_model = get_embedding_model("BAAI/bge-m3")
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )


def _build_where_filter(filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Constrói a cláusula where do ChromaDB de forma segura, ignorando chaves com valor None."""
    valid_conditions = []
    for key, val in filter_dict.items():
        if val is not None:
            valid_conditions.append({key: val})

    if not valid_conditions:
        return None
    if len(valid_conditions) == 1:
        return valid_conditions[0]
    return {"$and": valid_conditions}


def _sanitize_tool_text(text: str) -> str:
    """Sanitiza o texto retornado pelas ferramentas para evitar Jinja template syntax errors no LM Studio."""
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("{{", "{ {")
            .replace("}}", "} }")
            .replace("{%", "{ %")
            .replace("%}", "% }")
    )


@tool
def buscar_fala_transcricao(
    query: str,
    orador: Optional[str] = None,
    doc_id: Optional[int] = None
) -> str:

    """
    Busca trechos das falas registradas na Taquigrafia oficial das Audiências Públicas da Câmara dos Deputados.

    Use esta ferramenta quando precisar verificar o que um deputado, presidente ou convidado específico
    declarou durante uma audiência pública sobre determinado tema.

    Args:
        query (str): Pergunta ou texto temático para busca semântica na transcrição.
        orador (Optional[str]): Nome ou trecho do nome do orador/deputado para filtrar (ex: 'Lucas Redecker', 'Marcel van Hattem'). Padrão: None.
        doc_id (Optional[int]): ID numérico do documento/audiência pública (ex: 1). Padrão: None.

    Returns:
        str: Texto formatado com até 4 trechos de falas mais relevantes, seus metadados e scores de distância.
    """
    try:
        vs = _get_vector_store("transcricoes_db")
        where_clause = _build_where_filter({"orador": orador, "doc_id": doc_id})

        results = vs.similarity_search_with_score(
            query=query,
            k=4,
            filter=where_clause
        )

        if not results:
            return f"Nenhuma fala encontrada na transcrição para a consulta '{query}' com os filtros informados."

        output_lines = [f"=== RESULTADOS DA BUSCA EM TRANSCRIÇÕES (Query: '{query}') ==="]
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            doc_id_val = meta.get("doc_id", "N/A")
            orador_val = meta.get("orador", "Desconhecido")
            assunto_val = meta.get("assunto", "N/A")

            output_lines.append(
                f"\n--- Trecho {i} (Distância/Score: {score:.4f}) ---"
                f"\n[Doc ID]: {doc_id_val}"
                f"\n[Orador]: {_sanitize_tool_text(orador_val)}"
                f"\n[Assunto]: {_sanitize_tool_text(assunto_val)}"
                f"\n[Conteúdo]: {_sanitize_tool_text(doc.page_content)}"
            )

        return _sanitize_tool_text("\n".join(output_lines))

    except Exception as exc:
        logger.error(f"[agent_tools] Erro em buscar_fala_transcricao: {exc}")
        return f"Erro ao realizar busca vetorial na transcrição: {exc}"


@tool
def buscar_noticia_agencia_camara(
    query: str,
    doc_id: Optional[int] = None
) -> str:
    """
    Busca matérias e artigos jornalísticos publicados pela Agência Câmara cobrindo as audiências públicas.

    Use esta ferramenta para confrontar a cobertura jornalística com os fatos ocorridos ou verificar
    como determinado debate foi resumido pela imprensa oficial da Câmara.

    Args:
        query (str): Pergunta ou tema para busca semântica nas matérias jornalísticas.
        doc_id (Optional[int]): ID numérico do documento/audiência pública pareado (ex: 1). Padrão: None.

    Returns:
        str: Texto formatado com até 3 matérias jornalísticas mais relevantes e seus metadados.
    """
    try:
        vs = _get_vector_store("noticias_db")
        where_clause = _build_where_filter({"doc_id": doc_id})

        results = vs.similarity_search_with_score(
            query=query,
            k=3,
            filter=where_clause
        )

        if not results:
            return f"Nenhuma matéria jornalística encontrada para a consulta '{query}'."

        output_lines = [f"=== RESULTADOS DA BUSCA EM NOTÍCIAS (Query: '{query}') ==="]
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            doc_id_val = meta.get("doc_id", "N/A")
            assunto_val = meta.get("assunto", "N/A")

            output_lines.append(
                f"\n--- Matéria {i} (Distância/Score: {score:.4f}) ---"
                f"\n[Doc ID]: {doc_id_val}"
                f"\n[Assunto]: {_sanitize_tool_text(assunto_val)}"
                f"\n[Conteúdo]: {_sanitize_tool_text(doc.page_content)}"
            )

        return _sanitize_tool_text("\n".join(output_lines))

    except Exception as exc:
        logger.error(f"[agent_tools] Erro em buscar_noticia_agencia_camara: {exc}")
        return f"Erro ao realizar busca vetorial em notícias: {exc}"


@tool
def buscar_opiniao_estruturada_nli(
    query: str,
    apenas_alucinacoes: Optional[bool] = None
) -> str:
    """
    Busca opiniões estruturadas, posicionamentos e rótulos da frente NLI (Natural Language Inference).

    Use esta ferramenta para consultar a análise de fidelidade documental, opiniões de deputados/convidados
    e verificar se uma afirmação foi marcada como alucinação ou divergência factual.

    Args:
        query (str): Pergunta ou afirmação para busca semântica na base NLI.
        apenas_alucinacoes (Optional[bool]): Se True, filtra apenas registros onde verificacao_manual == False (alucinações/não-sustentadas). Padrão: None.

    Returns:
        str: Texto formatado com até 4 opiniões estruturadas, envolvidos, contextos e rótulos NLI.
    """
    try:
        vs = _get_vector_store("nli_metadados_db")
        
        where_dict = {}
        if apenas_alucinacoes is True:
            where_dict["verificacao_manual"] = False

        where_clause = _build_where_filter(where_dict)

        results = vs.similarity_search_with_score(
            query=query,
            k=4,
            filter=where_clause
        )

        if not results:
            return f"Nenhuma opinião NLI encontrada para a consulta '{query}' com os filtros aplicados."

        output_lines = [f"=== RESULTADOS DA BUSCA NLI (Query: '{query}') ==="]
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            doc_id_val = meta.get("doc_id", "N/A")
            nome_val = meta.get("envolvido_nome", "Desconhecido")
            cargo_val = meta.get("cargo", "N/A")
            verificacao_manual = meta.get("verificacao_manual", None)
            
            status_fidelidade = "Fiel/Sustentada (True)" if verificacao_manual is True else "Alucinação/Não-Sustentada (False)"

            output_lines.append(
                f"\n--- Opinião NLI {i} (Distância/Score: {score:.4f}) ---"
                f"\n[Doc ID]: {doc_id_val}"
                f"\n[Envolvido]: {_sanitize_tool_text(nome_val)} ({_sanitize_tool_text(cargo_val)})"
                f"\n[Status Verificação Manual]: {status_fidelidade}"
                f"\n[Conteúdo / Contexto]: {_sanitize_tool_text(doc.page_content)}"
            )

        return _sanitize_tool_text("\n".join(output_lines))

    except Exception as exc:
        logger.error(f"[agent_tools] Erro em buscar_opiniao_estruturada_nli: {exc}")
        return f"Erro ao realizar busca na base NLI: {exc}"

