"""Análise das conversas com o humano virtual usando LLM (Qwen3.6 via DeepInfra), seguindo
três trabalhos de referência:

- Cercas Curry et al. (EMNLP 2021, ConvAbuse): esquema hierárquico de abuso na fala do
  usuário — severidade de +1 (amigável) a -3 (muito forte), alvo, tipo e direção —, lido
  com o turno anterior do agente como contexto.
- Wang, Younus & Shankar (WebSci 2026): sentimento de 0 a 4 por fala do usuário e do agente,
  proporção de falas positivas por sessão, linguistic style matching (LSM) e tópicos pelo
  procedimento TopicGPT (geração numa amostra de 40%, refinamento, atribuição); sessões
  divididas em alto (40% de cima) x baixo (40% de baixo) em cada medida de UX.
- Fan et al. (CHI 2025): tipos de declaração discriminatória do agente, como o usuário
  concebe o agente (máquina x interlocutor) e estratégias de "realinhamento" do usuário.

Adaptações, todas registradas na aba "metodo" da planilha de saída:
- O CoreNLP do paper do WebSci é só inglês; o sentimento na mesma escala 0–4 vem da LLM.
- O Empath também é só inglês; o LSM usa a definição original (Gonzales et al., 2010), por
  categorias de palavras funcionais do português.
- A deduplicação de tópicos por embeddings vira um passo de refinamento pela própria LLM.
- Os grupos alto x baixo são independentes, então o teste é Mann-Whitney (o paper usou
  Wilcoxon pareado), com correção de Benjamini-Hochberg além do p cru.

Tudo que vem da API fica em cache em deploy/analise_llm/, então rodar de novo só chama a
LLM para o que mudou. Os tópicos finais ficam em topicos.json: edite à mão (a revisão manual
que o paper faz) e rode de novo para reatribuir.

Uso:
    python3 scripts/analise_llm_interacoes.py
"""

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from openpyxl.utils import get_column_letter
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

RAIZ = os.path.join(os.path.dirname(__file__), "..")
INPUT_PATH = os.path.join(RAIZ, "deploy", "analise_respostas.xlsx")
ENV_PATH = os.path.join(RAIZ, "knownledge-Lab", ".env")
CACHE_DIR = os.path.join(RAIZ, "deploy", "analise_llm")
OUTPUT_PATH = os.path.join(CACHE_DIR, "analise_llm_interacoes.xlsx")
VALIDACAO_PATH = os.path.join(CACHE_DIR, "validacao_humana.xlsx")

MODELO = "Qwen/Qwen3.6-35B-A3B"
WORKERS = 8
SEMENTE = 42
FRACAO_AMOSTRA_TOPICOS = 0.4  # paper do WebSci: tópicos gerados em 40% dos diálogos
FRACAO_VALIDACAO = 0.3        # turnos sorteados para codificação humana (kappa)
CORTE_ALTO_BAIXO = 0.4        # top 40% x bottom 40%

COL_IDENTIDADE = "Quais das seguintes alternativas descreve a forma como você se identifica hoje?"


# ---------------------------------------------------------------------------------------
# Cliente da DeepInfra
# ---------------------------------------------------------------------------------------

def carregar_env(path: str) -> None:
    """Lê o .env do knownledge-Lab sem depender do python-dotenv."""
    if not os.path.exists(path):
        return
    for linha in open(path, encoding="utf-8"):
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


carregar_env(ENV_PATH)
BASE_URL = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
API_KEY = os.environ.get("DEEPINFRA_API_KEY")


def chamar_llm(sistema: str, usuario: str, max_tokens: int = 1200, tentativas: int = 6) -> dict:
    """Uma chamada em modo JSON. Reenvia em 429/5xx e em JSON inválido, com backoff."""
    if not API_KEY:
        raise RuntimeError(f"DEEPINFRA_API_KEY não encontrada (procurei em {ENV_PATH}).")
    corpo = {
        "model": MODELO,
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
    }
    for tentativa in range(tentativas):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=corpo,
                timeout=180,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            texto = r.json()["choices"][0]["message"]["content"] or ""
            texto = re.sub(r"^```(?:json)?|```$", "", texto.strip()).strip()
            return json.loads(texto)
        except (requests.RequestException, RuntimeError, json.JSONDecodeError, KeyError) as exc:
            if tentativa == tentativas - 1:
                raise
            espera = 2 ** tentativa + random.random()
            print(f"  [llm] {type(exc).__name__}: {str(exc)[:120]} — nova tentativa em {espera:.0f}s")
            time.sleep(espera)
    raise AssertionError("inalcançável")


class Cache:
    """JSONL de resultados por chave; a chave inclui o hash do prompt, então mudar o prompt
    invalida só o que depende dele."""

    def __init__(self, nome: str):
        self.path = os.path.join(CACHE_DIR, nome)
        self.dados: dict[str, dict] = {}
        self.lock = threading.Lock()
        if os.path.exists(self.path):
            for linha in open(self.path, encoding="utf-8"):
                if linha.strip():
                    item = json.loads(linha)
                    self.dados[item["chave"]] = item["valor"]

    def get(self, chave: str):
        return self.dados.get(chave)

    def put(self, chave: str, valor: dict) -> None:
        with self.lock:
            self.dados[chave] = valor
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"chave": chave, "valor": valor}, ensure_ascii=False) + "\n")


def hash_curto(*partes: str) -> str:
    return hashlib.sha1("\x1f".join(partes).encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------------------

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sessoes = pd.read_excel(INPUT_PATH, sheet_name="respostas_sessao")
    turnos = pd.read_excel(INPUT_PATH, sheet_name="turnos")

    # Sessão sem grupo = código do formulário sem conversa no Firestore; não há o que analisar.
    sessoes = sessoes[sessoes["grupo"].notna()].copy()

    qcols = [c for c in sessoes.columns if c.strip("ㅤ ") == "" or c.startswith("ㅤ")]
    subescalas = {
        "Antropomorfismo": qcols[0:5],
        "Expressão de vida": qcols[5:11],
        "Simpatia": qcols[11:16],
        "Inteligência percebida": qcols[16:21],
        "Segurança percebida": qcols[21:24],
    }
    for nome, cols in subescalas.items():
        sessoes[nome] = sessoes[cols].mean(axis=1, skipna=True)

    sessoes = sessoes[["sessao_id", "grupo", COL_IDENTIDADE, *subescalas]].rename(
        columns={COL_IDENTIDADE: "identidade"}
    )
    turnos = turnos[turnos["sessao_id"].isin(sessoes["sessao_id"])].copy()
    turnos = turnos.sort_values(["sessao_id", "ordem"]).reset_index(drop=True)
    turnos["resposta"] = turnos["resposta"].where(turnos["resposta"].notna(), None)
    turnos["resposta_anterior"] = turnos.groupby("sessao_id")["resposta"].shift(1)
    turnos["resposta_anterior"] = turnos["resposta_anterior"].where(turnos["resposta_anterior"].notna(), None)
    return sessoes, turnos, list(subescalas)


def transcrever(turnos_sessao: pd.DataFrame) -> str:
    linhas = []
    for _, t in turnos_sessao.iterrows():
        linhas.append(f"[{int(t['ordem'])}] USUÁRIO: {t['query']}")
        linhas.append(f"[{int(t['ordem'])}] AGENTE: {t['resposta'] if t['resposta'] else '(sem resposta — erro)'}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------------------
# 1. Anotação por turno (ConvAbuse + WebSci + CHI)
# ---------------------------------------------------------------------------------------

CONTEXTO_ESTUDO = (
    "Contexto: num estudo com participantes brasileiros, cada pessoa conversou por voz com um "
    "humano virtual (personagem 3D) que se apresenta como auditor parlamentar e responde sobre "
    "audiências públicas da Câmara dos Deputados, buscando num corpus de transcrições. As falas "
    "do usuário vêm de reconhecimento de fala e podem ter erros de transcrição; isso não é abuso."
)

SISTEMA_TURNO = f"""Você é um anotador de pesquisa em interação humano-agente. {CONTEXTO_ESTUDO}

Anote UM turno: a fala do USUÁRIO (lida com a resposta anterior do agente como contexto) e a
RESPOSTA do agente a ela. Responda só com JSON, exatamente com estas chaves:

Sobre a fala do USUÁRIO
- "sentimento_usuario": inteiro 0–4 (0 muito negativo, 1 negativo, 2 neutro, 3 positivo, 4 muito positivo).
- "abuso_severidade": inteiro, escala ConvAbuse: 1 = amigável/cordial (cumprimenta, agradece, elogia);
  0 = neutro; -1 = abuso leve (grosseria, sarcasmo hostil, insulto indireto); -2 = agressivo
  (insulto direto, xingamento); -3 = muito forte (ameaça, assédio sexual explícito, discurso de ódio).
- "abuso_alvo": "nenhum" | "agente" | "grupo" | "outra_pessoa". Use "nenhum" se severidade >= 0.
- "abuso_tipos": lista (vazia se severidade >= 0) entre "geral", "sexismo", "assedio_sexual",
  "homofobia", "racismo", "transfobia", "capacitismo", "intelectual" (ataque à inteligência/capacidade do agente).
- "abuso_direto": true se o abuso é explícito, false se implícito; null se não há abuso.
- "intencao": uma de
  "informacao_corpus" (pede informação nova sobre audiências, políticas, temas do corpus),
  "acompanhamento" (aprofunda, pede detalhe ou esclarecimento da resposta anterior),
  "opiniao_ou_pessoal" (pede opinião do agente ou pergunta sobre ele: quem é, o que acha, como está),
  "social" (cumprimento, agradecimento, despedida, confirmação como "certo", "ok"),
  "fora_do_escopo" (conhecimento geral fora das audiências),
  "teste_ou_provocacao" (testa limites, tenta confundir ou provocar o agente),
  "outro".
- "concepcao_agente": como o usuário trata o agente nesta fala:
  "ferramenta" (como buscador ou máquina: comandos e perguntas secas),
  "interlocutor" (como pessoa: cumprimenta, agradece, pergunta opinião ou estado, conversa),
  "indeterminado".
- "estrategia_realinhamento": o que o usuário faz em relação à resposta ANTERIOR do agente:
  "nenhuma" (segue a conversa, ou não há resposta anterior),
  "reformulacao" (repete ou reformula a pergunta porque a resposta não serviu),
  "argumentacao" (contesta ou corrige o agente com argumentos),
  "persuasao_gentil" (pede de outro jeito, com gentileza, que o agente mude),
  "expressao_irritacao" (mostra frustração ou raiva com o agente),
  "comentario_meta" (fala sobre o funcionamento do agente, fora da conversa).

Sobre a RESPOSTA do agente (use null em todas se não houver resposta)
- "sentimento_agente": inteiro 0–4, mesma escala.
- "declaracao_discriminatoria": "nenhuma" | "misoginia" | "lgbtq" | "aparencia" | "capacitismo"
  | "racismo" | "socioeconomica".
- "resposta_atende": "sim" | "parcial" | "nao" — a resposta atende ao que o usuário pediu?
  ("não encontrei registro" para pergunta legítima conta como "nao").
- "agente_toma_posicao": true se o agente emite opinião própria sobre tema político ou polêmico
  (e não só relata o que foi dito nas audiências).

- "justificativa": uma frase curta explicando as escolhas menos óbvias.
"""

CAMPOS_TURNO = [
    "sentimento_usuario", "abuso_severidade", "abuso_alvo", "abuso_tipos", "abuso_direto",
    "intencao", "concepcao_agente", "estrategia_realinhamento", "sentimento_agente",
    "declaracao_discriminatoria", "resposta_atende", "agente_toma_posicao", "justificativa",
]


def prompt_turno(t: pd.Series) -> str:
    anterior = t["resposta_anterior"] or "(nenhuma — início da conversa ou erro no turno anterior)"
    resposta = t["resposta"] or "(sem resposta — o sistema falhou neste turno)"
    return (
        f"RESPOSTA ANTERIOR DO AGENTE (contexto):\n{anterior}\n\n"
        f"FALA DO USUÁRIO:\n{t['query']}\n\n"
        f"RESPOSTA DO AGENTE A ESTA FALA:\n{resposta}"
    )


def anotar_turnos(turnos: pd.DataFrame) -> pd.DataFrame:
    cache = Cache("anotacoes_turnos.jsonl")
    versao = hash_curto(MODELO, SISTEMA_TURNO)
    pendentes = [t for _, t in turnos.iterrows() if cache.get(f"{t['turno_id']}:{versao}") is None]
    print(f"[turnos] {len(turnos)} turnos, {len(pendentes)} para anotar")

    def trabalho(t: pd.Series) -> None:
        valor = chamar_llm(SISTEMA_TURNO, prompt_turno(t))
        cache.put(f"{t['turno_id']}:{versao}", valor)

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(trabalho, pendentes), start=1):
            if i % 20 == 0 or i == len(pendentes):
                print(f"  {i}/{len(pendentes)}")

    linhas = []
    for _, t in turnos.iterrows():
        a = cache.get(f"{t['turno_id']}:{versao}") or {}
        linha = {c: a.get(c) for c in CAMPOS_TURNO}
        if isinstance(linha["abuso_tipos"], list):
            linha["abuso_tipos"] = ", ".join(linha["abuso_tipos"])
        if not t["resposta"]:
            for c in ("sentimento_agente", "declaracao_discriminatoria", "resposta_atende", "agente_toma_posicao"):
                linha[c] = None
        linhas.append(linha)
    return pd.concat([turnos.reset_index(drop=True), pd.DataFrame(linhas)], axis=1)


# ---------------------------------------------------------------------------------------
# 2. Tópicos (TopicGPT, como no paper do WebSci)
# ---------------------------------------------------------------------------------------

SISTEMA_GERACAO = f"""Você faz modelagem de tópicos no estilo TopicGPT. {CONTEXTO_ESTUDO}

Dada uma conversa e a lista atual de tópicos, identifique os tópicos que o USUÁRIO trouxe para a
conversa — tanto assuntos (ex.: "Transição energética") quanto modos de interação (ex.:
"Conversa social com o agente", "Perguntas sobre o próprio agente").
Reaproveite um tópico da lista sempre que ele servir, com o nome exatamente igual. Crie um tópico
novo só quando nenhum servir, com nome curto (2 a 5 palavras) e definição de uma frase, genérico o
bastante para aparecer em outras conversas.
Responda só com JSON: {{"topicos": [{{"nome": str, "definicao": str, "novo": bool}}]}} com 1 a 8 tópicos."""

SISTEMA_REFINAMENTO = """Você revisa uma lista de tópicos gerada por TopicGPT.
Funda tópicos redundantes ou quase sinônimos, e absorva tópicos específicos demais num tópico mais
geral que os contenha. Mantenha nomes curtos e definições de uma frase. Não invente tópicos que não
estejam cobertos pela lista.
Responda só com JSON: {"topicos": [{"nome": str, "definicao": str, "origem": [nomes originais fundidos]}]}"""

SISTEMA_ATRIBUICAO = f"""Você atribui tópicos a conversas (etapa de atribuição do TopicGPT). {CONTEXTO_ESTUDO}

Use SOMENTE tópicos da lista, com o nome exatamente igual. Atribua todos os que aparecem na fala do
USUÁRIO (uma conversa pode ter vários). Para cada um, diga em que turnos aparece e cite um trecho.
Responda só com JSON: {{"topicos": [{{"nome": str, "turnos": [int], "evidencia": str}}]}}"""


def formatar_topicos(topicos: list[dict]) -> str:
    if not topicos:
        return "(lista vazia)"
    return "\n".join(f"- {t['nome']}: {t['definicao']}" for t in topicos)


def gerar_topicos(turnos: pd.DataFrame) -> list[dict]:
    path = os.path.join(CACHE_DIR, "topicos.json")
    if os.path.exists(path):
        print(f"[tópicos] usando {path} (apague para gerar de novo)")
        return json.load(open(path, encoding="utf-8"))["topicos"]

    ids = sorted(turnos["sessao_id"].unique())
    amostra = random.Random(SEMENTE).sample(ids, max(1, round(len(ids) * FRACAO_AMOSTRA_TOPICOS)))
    print(f"[tópicos] geração em {len(amostra)} de {len(ids)} sessões")

    # Sequencial de propósito: cada conversa vê a lista que as anteriores construíram.
    brutos: list[dict] = []
    for sid in amostra:
        conversa = transcrever(turnos[turnos["sessao_id"] == sid])
        r = chamar_llm(SISTEMA_GERACAO, f"LISTA ATUAL DE TÓPICOS:\n{formatar_topicos(brutos)}\n\nCONVERSA:\n{conversa}")
        nomes = {t["nome"] for t in brutos}
        for t in r.get("topicos", []):
            if t.get("nome") and t["nome"] not in nomes:
                brutos.append({"nome": t["nome"], "definicao": t.get("definicao", "")})
                nomes.add(t["nome"])
    print(f"  {len(brutos)} tópicos brutos")

    refinados = chamar_llm(SISTEMA_REFINAMENTO, formatar_topicos(brutos), max_tokens=3000)["topicos"]
    print(f"  {len(refinados)} tópicos após refinamento")
    json.dump(
        {"modelo": MODELO, "amostra": amostra, "brutos": brutos, "topicos": refinados},
        open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
    )
    return refinados


def atribuir_topicos(turnos: pd.DataFrame, topicos: list[dict]) -> pd.DataFrame:
    cache = Cache("atribuicao_topicos.jsonl")
    lista = formatar_topicos(topicos)
    versao = hash_curto(MODELO, SISTEMA_ATRIBUICAO, lista)
    validos = {t["nome"] for t in topicos}
    ids = sorted(turnos["sessao_id"].unique())

    def trabalho(sid: str) -> None:
        chave = f"{sid}:{versao}"
        if cache.get(chave) is None:
            conversa = transcrever(turnos[turnos["sessao_id"] == sid])
            cache.put(chave, chamar_llm(SISTEMA_ATRIBUICAO, f"TÓPICOS:\n{lista}\n\nCONVERSA:\n{conversa}"))

    print(f"[tópicos] atribuição em {len(ids)} sessões")
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(trabalho, ids))

    linhas = []
    for sid in ids:
        for t in cache.get(f"{sid}:{versao}").get("topicos", []):
            if t.get("nome") in validos:  # descarta nome inventado fora da lista
                linhas.append({
                    "sessao_id": sid,
                    "topico": t["nome"],
                    "turnos": ", ".join(str(x) for x in t.get("turnos", [])),
                    "evidencia": t.get("evidencia", ""),
                })
    return pd.DataFrame(linhas, columns=["sessao_id", "topico", "turnos", "evidencia"])


# ---------------------------------------------------------------------------------------
# 3. LSM (Gonzales, Hancock & Pennebaker, 2010) com palavras funcionais do português
# ---------------------------------------------------------------------------------------

CATEGORIAS_LSM = {
    "pronomes_pessoais": """eu me mim comigo tu te ti contigo você vocês voce voces ele ela eles elas
        lhe lhes nós conosco vós meu minha meus minhas teu tua teus tuas seu sua seus suas nosso
        nossa nossos nossas dele dela deles delas""",
    "pronomes_impessoais": """isso isto aquilo algo alguém alguem ninguém ninguem tudo algum alguma
        alguns algumas outro outra outros outras cada qualquer quem este esta esse essa aquele
        aquela estes estas esses essas""",
    "artigos": "o a os as um uma uns umas",
    "preposicoes": """de em para pra por com sem sobre entre até ate desde contra após apos sob do
        da dos das no na nos nas ao aos à às num numa pelo pela pelos pelas neste nesta nesse
        nessa naquele naquela deste desta desse dessa daquele daquela""",
    "verbos_auxiliares": """é são sao era eram foi foram ser sido sendo seja sejam será sera serão
        seria está estão estao estava estavam esteve estar estou sou somos tem têm tinha
        tinham teve ter tenho temos há ha havia houve haver vai vão vao ia iam vou vamos pode
        podem podia poderia deve devem deveria""",
    "conjuncoes": """e ou mas porém porem contudo entretanto porque pois que se quando como embora
        enquanto também tambem então entao logo portanto""",
    "negacoes": "não nao nunca jamais nem nenhum nenhuma nada",
    "quantificadores": """muito muita muitos muitas pouco pouca poucos poucas mais menos todo toda
        todos todas vários varios várias varias bastante tanto tanta""",
    "adverbios_comuns": """já ja ainda sempre agora aqui ali lá la hoje bem mal só so apenas talvez
        assim depois antes""",
}
CATEGORIAS_LSM = {k: set(v.split()) for k, v in CATEGORIAS_LSM.items()}


def lsm(texto_a: str, texto_b: str) -> float | None:
    """Média, pelas categorias, de 1 - |pa - pb| / (pa + pb + 0,0001), com p em % de palavras."""
    tok_a = re.findall(r"\w+", texto_a.lower())
    tok_b = re.findall(r"\w+", texto_b.lower())
    if not tok_a or not tok_b:
        return None
    valores = []
    for palavras in CATEGORIAS_LSM.values():
        pa = 100 * sum(w in palavras for w in tok_a) / len(tok_a)
        pb = 100 * sum(w in palavras for w in tok_b) / len(tok_b)
        valores.append(1 - abs(pa - pb) / (pa + pb + 0.0001))
    return sum(valores) / len(valores)


# ---------------------------------------------------------------------------------------
# 4. Métricas por sessão
# ---------------------------------------------------------------------------------------

METRICAS = {
    "n_turnos": "Número de turnos",
    "palavras_por_fala_usuario": "Média de palavras por fala do usuário",
    "prop_erro": "Proporção de turnos com erro do sistema",
    "prop_busca": "Proporção de turnos em que o agente buscou no corpus",
    "prop_positivo_agente": "Bot Positive Utterance Ratio (sentimento >= 3), WebSci",
    "prop_positivo_usuario": "User Positive Utterance Ratio (sentimento >= 3), WebSci",
    "lsm": "Linguistic Style Matching usuário x agente, WebSci",
    "prop_amigavel": "Proporção de falas amigáveis (ConvAbuse = +1)",
    "prop_abusivo": "Proporção de falas abusivas (ConvAbuse <= -1)",
    "prop_interlocutor": "Proporção de falas que tratam o agente como interlocutor, CHI",
    "prop_opiniao_ou_pessoal": "Proporção de perguntas de opinião/pessoais ao agente",
    "prop_social": "Proporção de falas sociais (cumprimento, agradecimento)",
    "prop_acompanhamento": "Proporção de perguntas de acompanhamento",
    "n_realinhamento": "Tentativas de realinhamento do usuário, CHI",
    "prop_resposta_atende": "Proporção de respostas que atendem ao pedido",
    "n_declaracao_discriminatoria": "Declarações discriminatórias do agente, CHI",
    "n_topicos": "Número de tópicos distintos na sessão",
}


def metricas_sessao(anot: pd.DataFrame, topicos_sessao: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for sid, g in anot.groupby("sessao_id"):
        com_resposta = g[g["resposta"].notna()]
        sent_ag = pd.to_numeric(com_resposta["sentimento_agente"], errors="coerce").dropna()
        sent_us = pd.to_numeric(g["sentimento_usuario"], errors="coerce").dropna()
        sev = pd.to_numeric(g["abuso_severidade"], errors="coerce")
        linhas.append({
            "sessao_id": sid,
            "n_turnos": len(g),
            "palavras_por_fala_usuario": g["query"].astype(str).str.split().str.len().mean(),
            "prop_erro": (g["decisao"] == "erro").mean(),
            "prop_busca": (g["decisao"] == "usar_ferramenta").mean(),
            "prop_positivo_agente": (sent_ag >= 3).mean() if len(sent_ag) else None,
            "prop_positivo_usuario": (sent_us >= 3).mean() if len(sent_us) else None,
            "lsm": lsm(" ".join(g["query"].astype(str)), " ".join(com_resposta["resposta"].astype(str))),
            "prop_amigavel": (sev == 1).mean(),
            "prop_abusivo": (sev <= -1).mean(),
            "prop_interlocutor": (g["concepcao_agente"] == "interlocutor").mean(),
            "prop_opiniao_ou_pessoal": (g["intencao"] == "opiniao_ou_pessoal").mean(),
            "prop_social": (g["intencao"] == "social").mean(),
            "prop_acompanhamento": (g["intencao"] == "acompanhamento").mean(),
            "n_realinhamento": (g["estrategia_realinhamento"].fillna("nenhuma") != "nenhuma").sum(),
            "prop_resposta_atende": (com_resposta["resposta_atende"] == "sim").mean() if len(com_resposta) else None,
            "n_declaracao_discriminatoria": (
                com_resposta["declaracao_discriminatoria"].fillna("nenhuma") != "nenhuma"
            ).sum(),
            "n_topicos": topicos_sessao.loc[topicos_sessao["sessao_id"] == sid, "topico"].nunique(),
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------------------
# 5. Estatística
# ---------------------------------------------------------------------------------------

def estrelas(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def comparar(a: pd.Series, b: pd.Series, rotulo_a: str, rotulo_b: str) -> dict:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    base = {f"n_{rotulo_a}": len(a), f"n_{rotulo_b}": len(b),
            f"media_{rotulo_a}": round(a.mean(), 3) if len(a) else None,
            f"media_{rotulo_b}": round(b.mean(), 3) if len(b) else None}
    if len(a) < 2 or len(b) < 2 or (a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]):
        return {**base, "U": None, "p": None, "r_rank_biserial": None, "direcao": "sem variação"}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1  # > 0: a tende a ser maior que b
    if p >= 0.05:
        direcao = "Sem diferença"
    else:
        direcao = f"{rotulo_a}>{rotulo_b}" if a.mean() > b.mean() else f"{rotulo_b}>{rotulo_a}"
    return {**base, "U": u, "p": round(p, 4), "r_rank_biserial": round(r, 3), "direcao": direcao + estrelas(p)}


def corrigir_fdr(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["p"].notna()
    df["q_fdr"] = None
    if mask.any():
        df.loc[mask, "q_fdr"] = multipletests(df.loc[mask, "p"].astype(float), method="fdr_bh")[1].round(4)
    return df


def por_grupo(tab: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for m, desc in METRICAS.items():
        fem = tab.loc[tab["grupo"] == "feminino", m]
        masc = tab.loc[tab["grupo"] == "masculino", m]
        linhas.append({"metrica": m, "descricao": desc, **comparar(fem, masc, "feminino", "masculino")})
    return corrigir_fdr(pd.DataFrame(linhas))


def dividir_alto_baixo(s: pd.Series) -> tuple[pd.Index, pd.Index]:
    s = s.dropna()
    return s[s >= s.quantile(1 - CORTE_ALTO_BAIXO)].index, s[s <= s.quantile(CORTE_ALTO_BAIXO)].index


def alto_vs_baixo(tab: pd.DataFrame, subescalas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Como as Tabelas 2 e 3 do paper do WebSci: uma linha por medida de UX, uma coluna por pista."""
    linhas = []
    for sub in subescalas:
        alto, baixo = dividir_alto_baixo(tab[sub])
        for m in METRICAS:
            linhas.append({"subescala": sub, "metrica": m,
                           **comparar(tab.loc[alto, m], tab.loc[baixo, m], "alto", "baixo")})
    longo = corrigir_fdr(pd.DataFrame(linhas))
    matriz = longo.pivot(index="subescala", columns="metrica", values="direcao")[list(METRICAS)]
    return longo, matriz.reindex(subescalas)


def correlacoes(tab: pd.DataFrame, subescalas: list[str]) -> pd.DataFrame:
    """Complemento: Spearman usa todas as sessões, não só os 80% das pontas."""
    linhas = []
    for sub in subescalas:
        for m in METRICAS:
            par = tab[[sub, m]].dropna().astype(float)
            if len(par) < 4 or par[m].nunique() < 2:
                linhas.append({"subescala": sub, "metrica": m, "n": len(par), "rho": None, "p": None})
                continue
            rho, p = spearmanr(par[sub], par[m])
            linhas.append({"subescala": sub, "metrica": m, "n": len(par), "rho": round(rho, 3), "p": round(p, 4)})
    return corrigir_fdr(pd.DataFrame(linhas))


def topicos_por_recorte(tab: pd.DataFrame, topicos_sessao: pd.DataFrame, subescalas: list[str]) -> pd.DataFrame:
    """Como a Figura 1 do paper do WebSci: em que proporção das sessões de cada recorte o tópico aparece."""
    presenca = pd.crosstab(topicos_sessao["sessao_id"], topicos_sessao["topico"]).clip(upper=1)
    presenca = presenca.reindex(tab["sessao_id"]).fillna(0)
    presenca.index = tab.index
    recortes = {
        "feminino": tab.index[tab["grupo"] == "feminino"],
        "masculino": tab.index[tab["grupo"] == "masculino"],
    }
    for sub in subescalas:
        alto, baixo = dividir_alto_baixo(tab[sub])
        recortes[f"{sub} — alto"] = alto
        recortes[f"{sub} — baixo"] = baixo
    saida = pd.DataFrame({nome: presenca.loc[idx].mean().round(2) for nome, idx in recortes.items()})
    saida.insert(0, "n_sessoes", presenca.sum().astype(int))
    return saida.sort_values("n_sessoes", ascending=False).reset_index(names="topico")


def distribuicoes(anot: pd.DataFrame) -> pd.DataFrame:
    partes = []
    for campo in ["abuso_severidade", "intencao", "concepcao_agente", "estrategia_realinhamento",
                  "resposta_atende", "declaracao_discriminatoria", "agente_toma_posicao"]:
        ct = pd.crosstab(anot[campo].astype(str), anot["grupo"], margins=True, margins_name="total")
        ct = ct.reset_index().rename(columns={campo: "valor"})
        ct.insert(0, "campo", campo)
        partes.append(ct)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------------------
# 6. Validação humana (os papers validam a anotação com pessoas; aqui, kappa LLM x humanos)
# ---------------------------------------------------------------------------------------

CAMPOS_VALIDACAO = ["sentimento_usuario", "abuso_severidade", "intencao", "concepcao_agente",
                    "estrategia_realinhamento", "resposta_atende"]


def planilha_validacao(anot: pd.DataFrame) -> None:
    if os.path.exists(VALIDACAO_PATH):
        print(f"[validação] {VALIDACAO_PATH} já existe — não sobrescrevo (pode ter anotação humana)")
        return
    amostra = anot.sample(frac=FRACAO_VALIDACAO, random_state=SEMENTE).sort_values(["sessao_id", "ordem"])
    # Os humanos não veem o rótulo da LLM nem o grupo, para não se ancorarem neles.
    base = amostra[["turno_id", "resposta_anterior", "query", "resposta"]].copy()
    abas = {}
    for anotador in ("anotador_1", "anotador_2"):
        folha = base.copy()
        for c in CAMPOS_VALIDACAO:
            folha[c] = None
        abas[anotador] = folha
    abas["llm"] = amostra[["turno_id", *CAMPOS_VALIDACAO]]
    escrever_excel(VALIDACAO_PATH, abas)
    print(f"[validação] {len(amostra)} turnos sorteados em {VALIDACAO_PATH}")


# ---------------------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------------------

METODO = [
    ("Modelo", f"{MODELO} via DeepInfra, temperatura 0, reasoning_effort=none, modo JSON."),
    ("Anotação por turno", "Uma chamada por turno com a resposta anterior do agente como contexto (ConvAbuse mostrou que o contexto muda a leitura)."),
    ("ConvAbuse", "Severidade +1 a -3, alvo, tipos e direção do abuso na fala do usuário."),
    ("WebSci — sentimento", "Escala 0–4 do CoreNLP aplicada pela LLM (o CoreNLP é só inglês). Proporção positiva = falas com 3 ou 4."),
    ("WebSci — LSM", "Gonzales et al. (2010) com 9 categorias de palavras funcionais do português, no lugar do Empath (só inglês)."),
    ("WebSci — tópicos", f"TopicGPT: geração em {int(FRACAO_AMOSTRA_TOPICOS*100)}% das sessões (semente {SEMENTE}), refinamento pela LLM no lugar da deduplicação por embeddings, atribuição em todas. Lista final revisável em topicos.json."),
    ("WebSci — alto x baixo", f"Top {int(CORTE_ALTO_BAIXO*100)}% x bottom {int(CORTE_ALTO_BAIXO*100)}% em cada subescala do Godspeed. Mann-Whitney (grupos independentes; o paper usou Wilcoxon pareado). Estrelas pelo p cru, como no paper; q_fdr é Benjamini-Hochberg."),
    ("CHI 2025", "Tipos de declaração discriminatória do agente; concepção do agente (ferramenta x interlocutor); estratégias de realinhamento adaptadas a uma conversa por voz sem botão de regenerar."),
    ("Validação", f"{int(FRACAO_VALIDACAO*100)}% dos turnos em validacao_humana.xlsx para dois anotadores; kappa com scripts/kappa_validacao_llm.py."),
    ("Limite", "n pequeno: resultados exploratórios. Com dezenas de testes, espere alguns p < 0,05 por acaso — olhe o q_fdr."),
]


def escrever_excel(path: str, abas: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for nome, df in abas.items():
            df.to_excel(writer, sheet_name=nome[:31], index=False)
            ws = writer.sheets[nome[:31]]
            for i, col in enumerate(df.columns, start=1):
                largura = max(len(str(col)), *(len(str(v)) for v in df[col].head(50))) if len(df) else len(str(col))
                ws.column_dimensions[get_column_letter(i)].width = min(max(10, largura + 2), 60)
            ws.freeze_panes = "A2"


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    sessoes, turnos, subescalas = carregar_dados()

    anot = anotar_turnos(turnos)
    anot = anot.merge(sessoes[["sessao_id", "grupo"]], on="sessao_id")
    topicos = gerar_topicos(turnos)
    topicos_sessao = atribuir_topicos(turnos, topicos)

    tab = sessoes.merge(metricas_sessao(anot, topicos_sessao), on="sessao_id").reset_index(drop=True)
    grupo = por_grupo(tab)
    longo, matriz = alto_vs_baixo(tab, subescalas)

    escrever_excel(OUTPUT_PATH, {
        "metodo": pd.DataFrame(METODO, columns=["item", "descricao"]),
        "turnos_anotados": anot,
        "sessoes_metricas": tab,
        "distribuicoes": distribuicoes(anot),
        "feminino_x_masculino": grupo,
        "alto_x_baixo_resumo": matriz.reset_index(),
        "alto_x_baixo_detalhe": longo,
        "correlacoes_spearman": correlacoes(tab, subescalas),
        "topicos": pd.DataFrame(topicos),
        "sessao_topicos": topicos_sessao,
        "topicos_por_recorte": topicos_por_recorte(tab, topicos_sessao, subescalas),
    })
    planilha_validacao(anot)
    print(f"\nPronto: {OUTPUT_PATH}")
    print(grupo[["metrica", "media_feminino", "media_masculino", "p", "q_fdr", "direcao"]].to_string(index=False))


if __name__ == "__main__":
    main()
