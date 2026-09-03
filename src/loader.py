"""
Módulo para download, verificação local e carregamento de dados do dataset PublicHearingBR.

Este módulo disponibiliza funções para gestão dos arquivos JSONL das frentes
LDS (Long Document Summarization) e NLI (Natural Language Inference).
"""

import json
import os
import urllib.request
from typing import Dict, List, Any, Optional
import pandas as pd


HF_REPO_ID = "unicamp-dl/PublicHearingBR"
LDS_FILENAME = "PublicHearingBR_LDS.jsonl"
NLI_FILENAME = "PublicHearingBR_NLI.jsonl"
BASE_HF_RAW_URL = "https://huggingface.co/datasets/unicamp-dl/PublicHearingBR/raw/main"


def _resolve_raw_dir(data_dir: Optional[str] = None) -> str:
    """Resolve o caminho do diretório de dados brutos procurando em locais padrão."""
    if data_dir is not None and os.path.exists(os.path.join(data_dir, LDS_FILENAME)):
        return data_dir

    candidates = [
        "notebooks/data/raw",
        "data/raw",
        "../data/raw",
        "data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, LDS_FILENAME)):
            return c

    return data_dir if data_dir is not None else "notebooks/data/raw"


def download_public_hearing_dataset(output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Baixa os arquivos do dataset PublicHearingBR do Hugging Face e os salva localmente.

    Verifica previamente se os arquivos já existem no diretório de destino
    para evitar downloads redundantes. Suporta arquivos armazenados via Git LFS.

    Args:
        output_dir (Optional[str]): Diretório local onde os arquivos JSONL serão salvos.
                                    Padrão: "notebooks/data/raw" (ou auto-resolvido).

    Returns:
        Dict[str, str]: Dicionário contendo os caminhos locais dos arquivos LDS e NLI.

    Raises:
        RuntimeError: Caso ocorra falha durante o download de algum arquivo.
    """
    import shutil
    from huggingface_hub import hf_hub_download

    target_dir = output_dir or _resolve_raw_dir()
    os.makedirs(target_dir, exist_ok=True)
    file_paths = {
        "lds": os.path.join(target_dir, LDS_FILENAME),
        "nli": os.path.join(target_dir, NLI_FILENAME),
    }

    files_to_download = {
        "lds": (LDS_FILENAME, file_paths["lds"]),
        "nli": (NLI_FILENAME, file_paths["nli"]),
    }

    for key, (filename, target_path) in files_to_download.items():
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            print(f"[loader] Arquivo existente e válido encontrado em: {target_path}. Download ignorado.")
            continue

        print(f"[loader] Obtendo {filename} via Hugging Face Hub (Git LFS)...")
        try:
            cached_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset"
            )
            shutil.copyfile(cached_path, target_path)
            print(f"[loader] Arquivo salvo com sucesso em: {target_path} ({os.path.getsize(target_path):,} bytes)")
        except Exception as exc:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise RuntimeError(f"Erro ao baixar {filename} do Hugging Face Hub: {exc}") from exc

    return file_paths


def _read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Lê um arquivo no formato JSONL e retorna uma lista de dicionários Python.

    Args:
        file_path (str): Caminho completo para o arquivo JSONL.

    Returns:
        List[Dict[str, Any]]: Lista contendo os objetos JSON lidos.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se houver erro de decodificação JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}. "
            "Execute a função download_public_hearing_dataset() primeiro."
        )

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as err:
                raise ValueError(f"Erro de parsing JSON no arquivo {file_path}, linha {line_num}: {err}") from err

    return records


def load_lds_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega os dados da frente LDS (Long Document Summarization).

    Estrutura retornada inclui transcrições de audiências públicas pareadas com matérias
    jornalísticas e metadados estruturados, enriquecidos com métricas descritivas.

    Args:
        data_dir (Optional[str]): Diretório onde o arquivo PublicHearingBR_LDS.jsonl está armazenado.
                                  Padrão: "notebooks/data/raw" (ou auto-resolvido).

    Returns:
        pd.DataFrame: DataFrame contendo id, transcrição, matéria, metadados e contagens.
    """
    target_dir = data_dir or _resolve_raw_dir()
    lds_file = os.path.join(target_dir, LDS_FILENAME)
    if not os.path.exists(lds_file):
        download_public_hearing_dataset(output_dir=target_dir)

    raw_records = _read_jsonl(lds_file)
    df = pd.DataFrame(raw_records)

    if not df.empty and "transcricao" in df.columns and "materia" in df.columns:
        df["char_count_transcricao"] = df["transcricao"].apply(lambda x: len(str(x)))
        df["word_count_transcricao"] = df["transcricao"].apply(lambda x: len(str(x).split()))
        df["char_count_materia"] = df["materia"].apply(lambda x: len(str(x)))
        df["word_count_materia"] = df["materia"].apply(lambda x: len(str(x).split()))
        df["compression_ratio"] = df["word_count_transcricao"] / df["word_count_materia"].replace(0, 1)

    return df


def load_nli_data(data_dir: Optional[str] = None, flatten: bool = True) -> pd.DataFrame:
    """
    Carrega os dados da frente NLI (Natural Language Inference).

    Permite carregar o dataset no formato bruto por amostra ou desaninhado (flatten),
    onde cada linha corresponde a uma opinião individual anotada com rótulos de alucinação.

    Args:
        data_dir (Optional[str]): Diretório onde o arquivo PublicHearingBR_NLI.jsonl está armazenado.
                                  Padrão: "notebooks/data/raw" (ou auto-resolvido).
        flatten (bool): Se True, desaninha as opiniões em linhas individuais. Padrão: True.

    Returns:
        pd.DataFrame: DataFrame com as opiniões e metadados NLI.
    """
    target_dir = data_dir or _resolve_raw_dir()
    nli_file = os.path.join(target_dir, NLI_FILENAME)
    if not os.path.exists(nli_file):
        download_public_hearing_dataset(output_dir=target_dir)

    raw_records = _read_jsonl(nli_file)

    if not flatten:
        return pd.DataFrame(raw_records)

    flattened_rows: List[Dict[str, Any]] = []

    for sample in raw_records:
        sample_id = sample.get("id")
        metadados = sample.get("metadados_extraidos", {})
        assunto = metadados.get("assunto", "")

        for envolvido in metadados.get("envolvidos", []):
            nome = envolvido.get("nome", "Desconhecido")
            cargo = envolvido.get("cargo", "Não Informado")

            for opiniao_dict in envolvido.get("opinioes", []):
                opiniao_texto = opiniao_dict.get("opiniao", "")
                chunks_proximos = opiniao_dict.get("chunks_proximos", [])
                verificacao = opiniao_dict.get("verificacao_alucinacao", {})

                verificacao_manual = verificacao.get("verificacao_manual")
                prompt_1 = verificacao.get("prompt_1_gpt-4o-mini-2024-07-18", {}).get("alucinacao")
                prompt_2 = verificacao.get("prompt_2_gpt-4o-mini-2024-07-18", {}).get("alucinacao")
                prompt_3 = verificacao.get("prompt_3_gpt-4o-mini-2024-07-18", {}).get("alucinacao")

                row = {
                    "sample_id": sample_id,
                    "assunto": assunto,
                    "nome_envolvido": nome,
                    "cargo_envolvido": cargo,
                    "opiniao": opiniao_texto,
                    "chunks_proximos": chunks_proximos,
                    "num_chunks_proximos": len(chunks_proximos),
                    "verificacao_manual_alucinacao": verificacao_manual,
                    "prompt_1_alucinacao": prompt_1,
                    "prompt_2_alucinacao": prompt_2,
                    "prompt_3_alucinacao": prompt_3,
                }
                flattened_rows.append(row)

    return pd.DataFrame(flattened_rows)
