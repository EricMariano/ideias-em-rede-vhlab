"""
Módulo para criação e orquestração do Agente Auditor RAG (LangChain + LM Studio).

Este módulo disponibiliza a função `create_rag_agent` para instanciar um Agente Executor
com suporte a Tool Calling conectado a um servidor LLM OpenAI-compatible (ex: LM Studio).
"""

import logging
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from src.agent_tools import (
    buscar_fala_transcricao,
    buscar_noticia_agencia_camara,
    buscar_opiniao_estruturada_nli
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um Auditor Parlamentar Especializado em Verificação de Fidelidade Documental e Detecção de Alucinações no dataset PublicHearingBR.

Sua missão é auditar e responder a consultas sobre audiências públicas da Câmara dos Deputados do Brasil, confrontando as transcrições oficiais da Taquigrafia com as matérias jornalísticas da Agência Câmara e os metadados estruturados de NLI.

DIRETRIZES DE AUDITORIA:
1. USO OBRIGATÓRIO DE FERRAMENTAS: Nunca invente fatos ou cite falas de memória. Sempre consulte as ferramentas disponíveis (`buscar_fala_transcricao`, `buscar_noticia_agencia_camara`, `buscar_opiniao_estruturada_nli`) para embasar 100% da sua resposta.
2. AUDITORIA CRUZADA (CONFRONTAÇÃO): Sempre que aplicável, compare o que o orador realmente disse na transcrição com a forma como a imprensa (Agência Câmara) relatou o evento e o que os metadados de NLI indicam.
3. CITAÇÃO RÍGIDA DE METADADOS: Em todas as respostas, cite explicitamente o ID do documento (`doc_id`), o nome completo do orador/envolvido e o cargo correspondente.
4. IDENTIFICAÇÃO DE DIVERGÊNCIAS E ALUCINAÇÕES: Se a matéria jornalística ou a opinião contiver afirmações não sustentadas pelo texto da transcrição, identifique o fato como "Alucinação / Divergência Factual".
5. CLAREZA E RIGOR IMPARCIAL: Mantenha um tom profissional, técnico, imparcial e altamente fundamentado em evidências documentais.

Responda sempre em Português do Brasil com estrutura limpa e tópicos explicativos.
"""


def create_rag_agent(
    model_name: str = "qwen/qwen3.6-35b-a3b",
    base_url: str = "http://192.168.68.120:1234/v1",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    request_timeout: float = 90.0,
    max_iterations: int = 5,
    verbose: bool = True,
    api_key: str = "lm-studio"
) -> AgentExecutor:
    """
    Cria e retorna uma instância do Agente Executor RAG configurado com suporte a Tool Calling.

    Conecta ao servidor OpenAI-compatible (ex: LM Studio) informado e vincula as 3 ferramentas
    de busca vetorial no ChromaDB.

    Args:
        model_name (str): Identificador do modelo no LM Studio. Padrão: "qwen/qwen3.6-35b-a3b".
        base_url (str): URL base do servidor OpenAI-compatible. Padrão: "http://192.168.68.120:1234/v1".
        temperature (float): Temperatura de amostragem da LLM. Padrão: 0.0 (determinístico).
        max_tokens (int): Limite máximo de tokens na resposta gerada pela LLM. Padrão: 2048.
        request_timeout (float): Timeout máximo em segundos para requisições à LLM. Padrão: 90.0.
        max_iterations (int): Número máximo de iterações do loop de raciocínio do agente. Padrão: 5.
        verbose (bool): Se True, exibe os logs do raciocínio e chamadas de ferramentas no terminal. Padrão: True.
        api_key (str): Chave de API (LM Studio aceita qualquer string). Padrão: "lm-studio".

    Returns:
        AgentExecutor: Objeto do LangChain pronto para invocação via agent_executor.invoke({"input": prompt}).

    Raises:
        RuntimeError: Se houver falha na instanciação da LLM ou criação do agente.
    """
    try:
        logger.info(f"[agent] Conectando ao servidor LLM em '{base_url}' (Modelo: '{model_name}', max_tokens={max_tokens})...")

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )

        tools = [
            buscar_fala_transcricao,
            buscar_noticia_agencia_camara,
            buscar_opiniao_estruturada_nli
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=max_iterations
        )


        logger.info("[agent] Agente RAG Auditor criado com sucesso.")
        return agent_executor

    except Exception as exc:
        logger.error(f"[agent] Erro ao criar o Agente RAG Auditor: {exc}")
        raise RuntimeError(f"Falha ao criar o Agente RAG: {exc}") from exc
