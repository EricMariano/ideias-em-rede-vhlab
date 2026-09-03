# MD0 - Estrutura dos Dados Baixados

> **Dataset de Referência:** [`unicamp-dl/PublicHearingBR`](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR)  
> **Domínio:** Sumarização de Documentos Longos (LDS) e Inferência em Linguagem Natural (NLI) no Português do Brasil  
> **Formato Original:** JSONL (JSON Lines)

---

## 1. Visão Geral do Dataset

O dataset **PublicHearingBR** foi desenvolvido especificamente para apoiar pesquisas em sumarização de documentos longos (*Long Document Summarization* - LDS) e inferência em linguagem natural (*Natural Language Inference* - NLI) no contexto do português brasileiro.

Ele é composto por dois subconjuntos principais de dados disponibilizados no formato JSONL:
1. `PublicHearingBR_LDS.jsonl`
2. `PublicHearingBR_NLI.jsonl`

Abaixo, detalha-se cada um desses datasets, seus schemas estruturais, descrições de campos e principais métricas estatísticas.

---

## 2. PublicHearingBR_LDS (Long Document Summarization)

Este subconjunto de dados é focado na tarefa de **sumarização de textos extensos**. Ele contém **206 amostras** que pareiam transcrições de audiências públicas da Câmara dos Deputados com matérias jornalísticas escritas por profissionais (atuando como resumos de referência) e metadados estruturados das opiniões defendidas.

### 2.1. Estrutura do Schema (JSONL)

Cada amostra do `PublicHearingBR_LDS.jsonl` possui a seguinte estrutura de dicionário com 4 atributos principais:

```json
{
  "id": 1,
  "transcricao": "Texto completo e extenso da transcrição da audiência pública...",
  "materia": "Texto da matéria jornalística correspondente publicada pela Agência Câmara...",
  "metadados": {
    "assunto": "Tema principal discutido na audiência pública",
    "envolvidos": [
      {
        "nome": "Nome do participante da audiência",
        "cargo": "Cargo e instituição/entidade do participante",
        "opinioes": [
          "Opinião detalhada defendida pelo envolvido 1",
          "Outra opinião ou posicionamento defendido pelo envolvido 1"
        ]
      }
    ]
  }
}
```

### 2.2. Descrição Detalhada dos Campos

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `id` | `int` | Identificador numérico sequencial da audiência pública (de 1 a 206). |
| `transcricao` | `str` | Documento longo de entrada contendo o texto bruto e integral da transcrição taquigráfica da audiência pública. |
| `materia` | `str` | Resumo textual de referência, correspondente à matéria jornalística produzida e publicada pela Agência Câmara de Notícias. |
| `metadados` | `dict` | Resumo estruturado em formato JSON contendo o tema e os atores envolvidos. |
| `metadados.assunto` | `str` | Pauta ou tópico principal debatido na audiência. |
| `metadados.envolvidos` | `list[dict]` | Lista com as pessoas cujas opiniões foram destacadas na matéria. Para cada pessoa, constam `nome`, `cargo` (se disponível) e uma lista de `opinioes` que representam seus posicionamentos detalhados retirados do texto. |

### 2.3. Estatísticas-Chave do LDS

* **Tamanho das Transcrições (`transcricao`):** Média de **18.102 palavras** por documento (podendo alcançar até **147.728 palavras**), evidenciando o grande desafio de processamento de contexto longo (*long-context*).
* **Tamanho das Notícias (`materia`):** Média de **627 palavras** por artigo.
* **Taxa de Compressão Média:** Cerca de **96%** (o resumo jornalístico representa, em média, apenas **4,20%** do volume da transcrição original).
* **Participantes e Opiniões:** Em média, cada amostra traz os posicionamentos de **5,2 indivíduos** e contém **10,7 opiniões estruturadas**.

---

## 3. PublicHearingBR_NLI (Natural Language Inference)

Este subconjunto de dados foi criado a partir de experimentos para investigar a ocorrência de **alucinações em modelos de IA generativa**. Ele contém **4.238 opiniões** extraídas de forma automática pelo ChatGPT. Para cada opinião, o dataset traz os trechos correspondentes da transcrição original e o rótulo de auditoria atestando se aquela opinião é logicamente fundamentada (inferível) ou se constitui uma alucinação factual.

### 3.1. Estrutura do Schema (JSONL)

Cada amostra do `PublicHearingBR_NLI.jsonl` possui a seguinte estrutura hierárquica:

```json
{
  "id": 1,
  "metadados_extraidos": {
    "assunto": "Tema principal da transcrição identificado pelo modelo",
    "envolvidos": [
      {
        "nome": "Nome do participante",
        "cargo": "Cargo do participante",
        "opinioes": [
          {
            "opiniao": "Texto da opinião extraída pelo sistema de IA",
            "chunks_proximos": [
              "Trecho de texto 1 recuperado da transcrição...",
              "Trecho de texto 2 recuperado da transcrição...",
              "Trecho de texto 3 recuperado da transcrição...",
              "Trecho de texto 4 recuperado da transcrição..."
            ],
            "verificacao_alucinacao": {
              "verificacao_manual": true, 
              "prompt_1_gpt-4o-mini-2024-07-18": {
                "alucinacao": false,
                "explicacao": "Raciocínio lógico do modelo..."
              },
              "prompt_2_gpt-4o-mini-2024-07-18": {
                "alucinacao": false,
                "trechos_para_basear_analise": ["Sentença específica..."],
                "explicacao": "Raciocínio lógico..."
              },
              "prompt_3_gpt-4o-mini-2024-07-18": {
                "alucinacao": false,
                "trechos_para_basear_analise": ["Sentença específica..."],
                "explicacao": "Raciocínio..."
              },
              "prompt_1_gpt-4o-2024-08-06": { "alucinacao": false, "explicacao": "..." },
              "prompt_2_gpt-4o-2024-08-06": { "alucinacao": false, "trechos_para_basear_analise": ["..."], "explicacao": "..." },
              "prompt_3_gpt-4o-2024-08-06": { "alucinacao": false, "trechos_para_basear_analise": ["..."], "explicacao": "..." },
              "prompt_1_deepseek-chat": { "alucinacao": false, "explicacao": "..." },
              "prompt_2_deepseek-chat": { "alucinacao": false, "trechos_para_basear_analise": ["..."], "explicacao": "..." },
              "prompt_3_deepseek-chat": { "alucinacao": false, "trechos_para_basear_analise": ["..."], "explicacao": "..." },
              "prompt_1_sabia-3.1-2025-05-08": { "alucinacao": false, "explicacao": "..." },
              "prompt_2_sabia-3.1-2025-05-08": { "alucinacao": false, "trechos_para_basear_analise": ["..."], "explicacao": "..." },
              "prompt_3_sabia-3.1-2025-05-08": { "alucinacao": false, "trechos_para_basear_analise": ["..."], "explicacao": "..." }
            }
          }
        ]
      }
    ]
  }
}
```

### 3.2. Descrição Detalhada dos Campos

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `id` | `int` | Código de referência associado à amostra de origem no dataset `PublicHearingBR_LDS`. |
| `metadados_extraidos` | `dict` | Contém a estrutura resultante da extração experimental conduzida via LLM. |
| `opiniao` | `str` | Texto exato da opinião extraída pelo modelo de IA. |
| `chunks_proximos` | `list[str]` | Lista com exatamente **4 blocos textuais** da transcrição original identificados (via similaridade de embeddings) como os mais semanticamente próximos e relevantes à opinião gerada. |
| `verificacao_alucinacao` | `dict` | Concentra as validações sobre a fidelidade da opinião em relação aos chunks. |
| `verificacao_manual` | `bool` | **Ground Truth Humano:** Marcação realizada por especialistas humanos (revisores consultores da Câmara dos Deputados) validando se a opinião pode ser inferida dos 4 chunks (`true` = válida / fiel; `false` = alucinação / não inferível). |
| *Prompts por Modelo* | `dict` | Resultados dos testes automatizados de detecção de alucinação utilizando juízes LLM (**GPT-4o mini, GPT-4o, DeepSeek-V3, Sabiá-3.1**) sob 3 estratégias de prompt. Cada um retorna `alucinacao` (bool), `explicacao` (string) e `trechos_para_basear_analise` (quando aplicável). |

### 3.3. Estatísticas-Chave do NLI

* **Total de Opiniões Avaliadas:** **4.238 opiniões**.
* **Opiniões Válidas (Anotação Manual):** **3.734 opiniões** (**88,11%** do total) confirmadas como inferíveis diretamente a partir dos textos.
* **Alucinações Confirmadas:** **504 opiniões** (**11,89%** do total) catalogadas como alucinações de conteúdo ou erros de atribuição de fala.
