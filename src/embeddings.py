"""
Módulo para inicialização e gestão do modelo de embeddings multilíngue (BAAI/bge-m3).

Este módulo fornece acesso seguro ao modelo de embedding da HuggingFace configurado
para execução em GPU ou CPU de acordo com a disponibilidade do ambiente.
"""

import logging
from functools import lru_cache
from typing import Optional
import torch

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = "BAAI/bge-m3",
    device: Optional[str] = None,
    encode_batch_size: int = 32
) -> HuggingFaceEmbeddings:
    """
    Instancia e retorna o modelo de embedding multilíngue `BAAI/bge-m3` via HuggingFaceEmbeddings.

    Detecta automaticamente se a GPU (CUDA) possui VRAM suficiente livre (mínimo de 3.5 GB para evitar
    CUDA Out Of Memory com BAAI/bge-m3); caso contrário, reverte para a CPU de forma segura.

    Prioriza o carregamento 100% offline (local_files_only=True) para evitar timeouts de rede.
    Caso o modelo não esteja em cache local, realiza o download automaticamente do HuggingFace Hub.

    Args:
        model_name (str): Nome ou caminho do modelo no Hugging Face Hub. Padrão: "BAAI/bge-m3".
        device (Optional[str]): Dispositivo de execução ('cuda', 'cpu', 'mps' ou None para auto-detecção).
        encode_batch_size (int): Tamanho do lote interno de inferência do SentenceTransformers. Padrão: 32.

    Returns:
        HuggingFaceEmbeddings: Instância do modelo de embedding pronta para codificação.

    Raises:
        RuntimeError: Se houver falha ao carregar o modelo de embedding.
    """
    try:
        if device is None:
            if torch.cuda.is_available():
                free_mem, _ = torch.cuda.mem_get_info()
                free_gb = free_mem / (1024 ** 3)
                if free_gb >= 3.5:
                    device = "cuda"
                    torch.cuda.empty_cache()
                else:
                    logger.warning(
                        f"[embeddings] VRAM livre na GPU ({free_gb:.2f} GB) insuficiente para {model_name} "
                        "(mínimo recomendado: 3.5 GB). Revertendo com segurança para CPU."
                    )
                    device = "cpu"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        encode_kwargs = {"normalize_embeddings": True, "batch_size": encode_batch_size}

        # 1ª Tentativa: Tenta carregar 100% offline a partir do cache local (evita checagens de rede no HF Hub)
        try:
            model_kwargs_local = {"device": device, "local_files_only": True}
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs_local,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"[embeddings] Modelo '{model_name}' carregado instantaneamente via cache local no dispositivo '{device}'.")
            return embedding_model
        except Exception:
            # 2ª Tentativa (Fallback): Se o modelo não existir localmente, conecta à internet para baixar
            logger.info(f"[embeddings] Modelo '{model_name}' não encontrado localmente. Conectando ao HuggingFace Hub para download...")
            model_kwargs_remote = {"device": device}
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs_remote,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"[embeddings] Modelo '{model_name}' baixado e inicializado com sucesso no dispositivo '{device}'.")
            return embedding_model

    except Exception as exc:
        logger.error(f"[embeddings] Erro ao carregar o modelo de embedding '{model_name}': {exc}")
        raise RuntimeError(f"Falha ao carregar o modelo de embedding '{model_name}': {exc}") from exc


