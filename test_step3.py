"""
Script CLI para validação técnica e testes unitários/integrativos do Subpasso 3.1.

Realiza o teste unitário offline das ferramentas (@tool) no ChromaDB e tenta
o teste integrativo de Tool Calling com a LLM no LM Studio em http://192.168.68.120:1234/v1.
"""

import sys
import logging
import urllib.request
import urllib.error
from src.agent_tools import (
    buscar_fala_transcricao,
    buscar_noticia_agencia_camara,
    buscar_opiniao_estruturada_nli
)
from src.agent import create_rag_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_unit_tools():
    print("=" * 80)
    print("🧪 1. TESTE UNITÁRIO DE FERRAMENTAS (@tool) - OFFLINE (CHROMADB)")
    print("=" * 80)

    print("\n🔹 Teste 1.1: `buscar_fala_transcricao`...")
    res_1 = buscar_fala_transcricao.invoke({
        "query": "censura e liberdade de expressão",
        "orador": "Lucas Redecker",
        "doc_id": 1
    })
    print(f"   [OK] Retorno ({len(res_1)} chars):")
    print(f"   {res_1[:300]}...\n")

    print("🔹 Teste 1.2: `buscar_noticia_agencia_camara`...")
    res_2 = buscar_noticia_agencia_camara.invoke({
        "query": "liberdade de expressão redes sociais",
        "doc_id": 1
    })
    print(f"   [OK] Retorno ({len(res_2)} chars):")
    print(f"   {res_2[:300]}...\n")

    print("🔹 Teste 1.3: `buscar_opiniao_estruturada_nli`...")
    res_3 = buscar_opiniao_estruturada_nli.invoke({
        "query": "alexandre de moraes censura",
        "apenas_alucinacoes": False
    })
    print(f"   [OK] Retorno ({len(res_3)} chars):")
    print(f"   {res_3[:300]}...\n")

    print("✅ Todos os testes unitários offline das ferramentas foram concluídos com SUCESSO!")


def test_integrative_agent():
    print("\n" + "=" * 80)
    print("🤖 2. TESTE INTEGRATIVO DE TOOL CALLING (LM STUDIO REMOTO)")
    print("=" * 80)

    base_url = "http://192.168.68.120:1234/v1"
    models_url = f"{base_url}/models"
    
    print(f"⏳ Verificando conectividade com o LM Studio em '{base_url}'...")

    server_online = False
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "PythonTest"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                server_online = True
                print("   [ONLINE] Servidor LM Studio encontrado e responsivo!")
    except Exception as err:
        print(f"   [OFFLINE] Servidor LM Studio indisponível no IP '192.168.68.120:1234' ({err}).")

    if not server_online:
        print("\n⚠️ O teste integrativo online foi ignorado pois o servidor remoto do LM Studio está offline.")
        print("💡 Para executar o teste online com a LLM:")
        print("   1. Certifique-se de que o LM Studio está rodando no servidor em http://192.168.68.120:1234")
        print("   2. Carregue o modelo 'qwen/qwen3.6-35b-a3b' (ou similar compatível com Tool Calling)")
        print("   3. Re-execute o script: python test_step3.py")
        return

    try:
        agent_executor = create_rag_agent(
            model_name="qwen/qwen3.6-35b-a3b",
            base_url=base_url,
            temperature=0.0
        )

        query = "Verifique o que o Deputado Lucas Redecker disse na audiência pública 1 e compare com as matérias jornalísticas disponíveis para saber se houve fidelidade."
        print(f"\n❓ Pergunta enviada ao Agente Auditor:\n   '{query}'\n")

        response = agent_executor.invoke({"input": query})

        print("\n" + "=" * 80)
        print("💬 RESPOSTA FINAL FUNDAMENTADA DO AGENTE AUDITOR:")
        print("=" * 80)
        print(response.get("output", "Sem resposta gerada."))
        print("=" * 80)

    except Exception as exc:
        print(f"\n❌ Erro durante o teste integrativo do Agente: {exc}")


if __name__ == "__main__":
    test_unit_tools()
    test_integrative_agent()
