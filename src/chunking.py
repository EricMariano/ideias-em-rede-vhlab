"""
Módulo para estratégias de fatiamento (chunking) da Abordagem Híbrida do pipeline RAG.

Este módulo disponibiliza funções para fatiar transcrições da Taquigrafia da Câmara por orador,
matérias jornalísticas por tamanho de caractere/token e registros desaninhados da frente NLI.
"""

import re
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Expressão regular robusta para detectar cabeçalhos de oradores na Taquigrafia da Câmara dos Deputados.
# Suporta prefixos (O SR., A SRA., O CONVIDADO, etc.), nomes, cargos e blocos parentéticos contendo hífens (ex: Bloco/PSDB - RS).
SPEAKER_HEADER_RE = re.compile(
    r'(?:\n|^)\s*'
    r'((?:O\s+SR\.|A\s+SRA\.|O\s+CONVIDADO|A\s+CONVIDADA|O\s+MINISTRO|A\s+MINISTRA|O\s+DEPUTADO|A\s+DEPUTADA|SR\.|SRA\.)'
    r'[^(\n\-:]*(?:\([^)]*\)[^(\n\-:]*)*)'
    r'\s*[\-:]\s*',
    re.IGNORECASE
)


def _sanitize_metadata_value(val: Any) -> Any:
    """
    Sanitiza valores de metadados para compatibilidade com o ChromaDB.
    
    Converte nulos (None/NaN) em strings vazias ou booleans para evitar exceções do Vector Store.
    """
    if pd.isna(val) or val is None:
        return ""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float, str)):
        return val
    return str(val)


def chunk_transcriptions_by_speaker(
    df_lds: pd.DataFrame,
    max_tokens: int = 800,
    overlap: int = 150
) -> List[Document]:
    """
    Fatia transcrições de audiências públicas agrupando o texto por orador/deputado.

    Utiliza Expressões Regulares para identificar falas individuais da Taquigrafia da Câmara.
    Se a fala de um único orador exceder `max_tokens`, aplica um fallback com `RecursiveCharacterTextSplitter`.

    Args:
        df_lds (pd.DataFrame): DataFrame contendo os dados brutos da frente LDS (deve possuir 'id', 'transcricao' e 'metadados').
        max_tokens (int): Tamanho máximo aproximado em tokens por chunk. Padrão: 800.
        overlap (int): Sobreposição aproximada em tokens para o splitter de fallback. Padrão: 150.

    Returns:
        List[Document]: Lista de objetos Document do LangChain com metadados estruturados.
    """
    documents: List[Document] = []
    
    # Aproximação em caracteres (~4 caracteres por token em português)
    approx_chunk_size = max_tokens * 4
    approx_overlap = overlap * 4

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=approx_chunk_size,
        chunk_overlap=approx_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    for idx, row in df_lds.iterrows():
        doc_id = _sanitize_metadata_value(row.get("id"))
        transcricao = str(row.get("transcricao", "")).strip()
        
        metadados_dict = row.get("metadados") if isinstance(row.get("metadados"), dict) else {}
        assunto = _sanitize_metadata_value(metadados_dict.get("assunto", ""))

        if not transcricao:
            continue

        matches = list(SPEAKER_HEADER_RE.finditer(transcricao))

        if not matches:
            # Fallback global para transcrições sem marcadores formais de orador
            sub_chunks = fallback_splitter.split_text(transcricao)
            for sub_idx, sub_text in enumerate(sub_chunks):
                doc = Document(
                    page_content=sub_text,
                    metadata={
                        "doc_id": doc_id,
                        "orador": "Desconhecido",
                        "assunto": assunto,
                        "tipo": "transcricao"
                    }
                )
                documents.append(doc)
            continue

        # Processa cada fala de orador identificada
        for i, match in enumerate(matches):
            start_content = match.end()
            end_content = matches[i + 1].start() if i + 1 < len(matches) else len(transcricao)
            
            header_str = match.group(1).strip()
            speech_content = transcricao[start_content:end_content].strip()
            
            if not speech_content:
                continue

            full_speech = f"{header_str} - {speech_content}"
            
            # Estimativa de tokens
            estimated_tokens = len(full_speech) // 4
            
            if estimated_tokens > max_tokens:
                sub_chunks = fallback_splitter.split_text(full_speech)
                for sub_text in sub_chunks:
                    documents.append(
                        Document(
                            page_content=sub_text,
                            metadata={
                                "doc_id": doc_id,
                                "orador": header_str,
                                "assunto": assunto,
                                "tipo": "transcricao"
                            }
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=full_speech,
                        metadata={
                            "doc_id": doc_id,
                            "orador": header_str,
                            "assunto": assunto,
                            "tipo": "transcricao"
                        }
                    )
                )

    return documents


def chunk_news_articles(
    df_lds: pd.DataFrame,
    chunk_size: int = 1000,
    overlap: int = 150
) -> List[Document]:
    """
    Fatia matérias jornalísticas de audiências públicas em blocos de texto usando RecursiveCharacterTextSplitter.

    Args:
        df_lds (pd.DataFrame): DataFrame contendo os dados da frente LDS (deve possuir 'id', 'materia' e 'metadados').
        chunk_size (int): Tamanho do chunk em caracteres. Padrão: 1000.
        overlap (int): Sobreposição em caracteres entre chunks. Padrão: 150.

    Returns:
        List[Document]: Lista de documentos LangChain com metadados da matéria jornalística.
    """
    documents: List[Document] = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    for idx, row in df_lds.iterrows():
        doc_id = _sanitize_metadata_value(row.get("id"))
        materia_text = str(row.get("materia", "")).strip()

        metadados_dict = row.get("metadados") if isinstance(row.get("metadados"), dict) else {}
        assunto = _sanitize_metadata_value(metadados_dict.get("assunto", ""))

        if not materia_text:
            continue

        chunks = text_splitter.split_text(materia_text)

        for chunk_text in chunks:
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "assunto": assunto,
                        "tipo": "noticia"
                    }
                )
            )

    return documents


def chunk_nli_records(df_nli: pd.DataFrame) -> List[Document]:
    """
    Converte registros desaninhados do NLI (opiniões e chunks próximos) em documentos LangChain.

    Args:
        df_nli (pd.DataFrame): DataFrame desaninhado da frente NLI (gerado por load_nli_data(flatten=True)).

    Returns:
        List[Document]: Lista de documentos LangChain representando opiniões NLI.
    """
    documents: List[Document] = []

    for idx, row in df_nli.iterrows():
        doc_id = _sanitize_metadata_value(row.get("sample_id"))
        nome = _sanitize_metadata_value(row.get("nome_envolvido"))
        cargo = _sanitize_metadata_value(row.get("cargo_envolvido"))
        opiniao = str(row.get("opiniao", "")).strip()
        
        raw_manual = row.get("verificacao_manual_alucinacao")
        verificacao_manual = bool(raw_manual) if pd.notna(raw_manual) and raw_manual is not None else False

        chunks_proximos = row.get("chunks_proximos", [])
        if isinstance(chunks_proximos, list):
            chunks_text = "\n---\n".join([str(c).strip() for c in chunks_proximos if c])
        else:
            chunks_text = str(chunks_proximos).strip()

        page_content = f"Opinião ({nome} - {cargo}): {opiniao}"
        if chunks_text:
            page_content += f"\n\nContexto (Chunks Próximos):\n{chunks_text}"

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "doc_id": doc_id,
                    "envolvido_nome": nome,
                    "cargo": cargo,
                    "verificacao_manual": verificacao_manual,
                    "tipo": "nli_opinion"
                }
            )
        )

    return documents
