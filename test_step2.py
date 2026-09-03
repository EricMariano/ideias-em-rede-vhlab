"""
Script CLI para validação técnica e teste rápido do Subpasso 2.1 do pipeline RAG.

Carrega 1 amostra da frente LDS e 1 amostra da frente NLI via `src.loader`,
executa o fatiamento em `src.chunking` e exibe o número de chunks gerados
e os metadados de 1 chunk de cada tipo (transcrição, notícia, nli_opinion).
"""

import sys
import logging
from src.loader import load_lds_data, load_nli_data
from src.chunking import (
    chunk_transcriptions_by_speaker,
    chunk_news_articles,
    chunk_nli_records
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_validation():
    print("=" * 80)
    print("🧪 INICIANDO TESTE DE VALIDAÇÃO CLI - SUBPASSO 2.1 (RAG CORE)")
    print("=" * 80)

    # 1. Carregar 1 amostra de LDS e NLI via src.loader
    print("\n1. Carregando dados via src.loader...")
    df_lds_full = load_lds_data()
    df_nli_full = load_nli_data(flatten=True)

    df_lds_sample = df_lds_full.head(1).copy()
    
    # Amostra NLI correspondente à primeira amostra LDS
    sample_id = df_lds_sample.iloc[0]["id"]
    df_nli_sample = df_nli_full[df_nli_full["sample_id"] == sample_id].copy()
    if df_nli_sample.empty:
        df_nli_sample = df_nli_full.head(3).copy()

    print(f"   [OK] Amostra LDS carregada: {len(df_lds_sample)} registro(s) (ID={sample_id})")
    print(f"   [OK] Amostra NLI carregada: {len(df_nli_sample)} registro(s)")

    # 2. Executar Fatiamento (Chunking)
    print("\n2. Executando fatiamento em src.chunking...")
    
    transcription_chunks = chunk_transcriptions_by_speaker(df_lds_sample, max_tokens=800, overlap=150)
    news_chunks = chunk_news_articles(df_lds_sample, chunk_size=1000, overlap=150)
    nli_chunks = chunk_nli_records(df_nli_sample)

    print(f"   [RESULTADO] Chunks de Transcrição (por Orador): {len(transcription_chunks)}")
    print(f"   [RESULTADO] Chunks de Matéria Jornalística: {len(news_chunks)}")
    print(f"   [RESULTADO] Chunks de NLI (Opiniões): {len(nli_chunks)}")

    # 3. Exibir Metadados de 1 chunk de cada tipo
    print("\n" + "=" * 80)
    print("📋 INSPEÇÃO DE METADADOS DOS CHUNKS GERADOS")
    print("=" * 80)

    if transcription_chunks:
        sample_doc = transcription_chunks[0]
        print("\n🔹 Chunk Tipo 'transcricao' (Amostra 1):")
        print(f"   • Metadados: {sample_doc.metadata}")
        print(f"   • Prévia Conteúdo ({len(sample_doc.page_content)} chars):\n     {sample_doc.page_content[:200]}...")

    if news_chunks:
        sample_doc = news_chunks[0]
        print("\n🔹 Chunk Tipo 'noticia' (Amostra 1):")
        print(f"   • Metadados: {sample_doc.metadata}")
        print(f"   • Prévia Conteúdo ({len(sample_doc.page_content)} chars):\n     {sample_doc.page_content[:200]}...")

    if nli_chunks:
        sample_doc = nli_chunks[0]
        print("\n🔹 Chunk Tipo 'nli_opinion' (Amostra 1):")
        print(f"   • Metadados: {sample_doc.metadata}")
        print(f"   • Prévia Conteúdo ({len(sample_doc.page_content)} chars):\n     {sample_doc.page_content[:200]}...")

    print("\n" + "=" * 80)
    print("✅ TESTE DE VALIDAÇÃO DO SUBPASSO 2.1 CONCLUÍDO COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    run_validation()
