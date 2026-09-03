"""
Módulo para gestão da Vector Store (ChromaDB) e indexação em lote.

Este módulo disponibiliza funções para obtenção do cliente persistente do ChromaDB
e indexação de documentos em lotes configuráveis com tratamento de erros.
"""

import os
import logging
from typing import List, Optional, Any
import chromadb
from tqdm.auto import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


def _resolve_chroma_dir(persist_directory: Optional[str] = None) -> str:
    """Resolve o diretório do ChromaDB priorizando onde o banco sqlite3 já existe."""
    if persist_directory is not None and os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        return persist_directory

    candidates = [
        "notebooks/data/processed/chroma_db",
        "data/processed/chroma_db",
        "../data/processed/chroma_db",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "chroma.sqlite3")):
            return c

    return persist_directory if persist_directory is not None else "notebooks/data/processed/chroma_db"


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    """
    Retorna o cliente persistente do ChromaDB configurado no diretório informado.

    Args:
        persist_directory (Optional[str]): Diretório para persistência do banco de dados vetorial.
                                           Padrão: "notebooks/data/processed/chroma_db" (ou auto-resolvido).

    Returns:
        chromadb.PersistentClient: Instância do cliente persistente do ChromaDB.

    Raises:
        RuntimeError: Se ocorrer erro na criação ou conexão com o ChromaDB.
    """
    try:
        target_dir = persist_directory or _resolve_chroma_dir()
        os.makedirs(target_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=target_dir)
        logger.info(f"[vector_store] Cliente ChromaDB inicializado em '{target_dir}'.")
        return client
    except Exception as exc:
        logger.error(f"[vector_store] Erro ao conectar ao ChromaDB em '{persist_directory}': {exc}")
        raise RuntimeError(f"Erro ao inicializar cliente ChromaDB: {exc}") from exc


def index_documents_in_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    documents: List[Document],
    embedding_model: Any,
    batch_size: int = 100,
    overwrite: bool = False
) -> Chroma:
    """
    Cria ou obtém uma coleção no ChromaDB e insere os documentos em lotes (batches) configuráveis com tqdm.

    Se a coleção já possuir os documentos completos e overwrite=False, reutiliza a coleção existente
    sem re-calcular embeddings, garantindo alta performance e idempotência.

    Args:
        client (chromadb.PersistentClient): Cliente persistente do ChromaDB.
        collection_name (str): Nome da coleção no ChromaDB.
        documents (List[Document]): Lista de documentos LangChain a serem indexados.
        embedding_model (Any): Instância do modelo de embedding (ex: HuggingFaceEmbeddings).
        batch_size (int): Quantidade de documentos por lote de inserção. Padrão: 100.
        overwrite (bool): Se True, deleta e recria a coleção antes da indexação. Padrão: False.

    Returns:
        Chroma: Objeto VectorStore do LangChain associado à coleção indexada.

    Raises:
        ValueError: Se a lista de documentos estiver vazia.
        RuntimeError: Se houver erro durante o processo de indexação.
    """
    if not documents:
        raise ValueError("A lista de documentos fornecida para indexação está vazia.")

    try:
        total_docs = len(documents)
        
        if overwrite:
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"[vector_store] Coleção '{collection_name}' removida para re-indexação.")
            except Exception:
                pass

        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )

        existing_count = vector_store._collection.count()
        if existing_count >= total_docs and not overwrite:
            print(f"[vector_store] Coleção '{collection_name}' já contém {existing_count} documentos. Reutilizando dados persitidos.")
            return vector_store

        if existing_count > 0 and existing_count < total_docs and not overwrite:
            print(f"[vector_store] Coleção '{collection_name}' incompleta ({existing_count}/{total_docs} docs). Recriando coleção do zero...")
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_model
            )

        print(f"[vector_store] Indexando {total_docs} documentos na coleção '{collection_name}' em lotes de {batch_size}...")

        num_batches = (total_docs + batch_size - 1) // batch_size
        for batch_num in tqdm(range(num_batches), desc=f"Indexando {collection_name}"):
            i = batch_num * batch_size
            batch = documents[i : i + batch_size]
            
            ids = [
                f"{doc.metadata.get('tipo', 'doc')}_{doc.metadata.get('doc_id', '0')}_b{batch_num}_{idx}"
                for idx, doc in enumerate(batch)
            ]
            
            vector_store.add_documents(documents=batch, ids=ids)
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        final_count = vector_store._collection.count()
        print(f"[vector_store] Indexação concluída. Total de documentos na coleção '{collection_name}': {final_count}")
        return vector_store

    except Exception as exc:
        logger.error(f"[vector_store] Erro ao indexar documentos na coleção '{collection_name}': {exc}")
        raise RuntimeError(f"Falha na indexação dos documentos no ChromaDB: {exc}") from exc
