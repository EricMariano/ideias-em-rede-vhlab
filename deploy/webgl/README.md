# Deploy (Railway) — build WebGL + Piper TTS

Uma imagem, um serviço, um domínio: o Caddy serve o build WebGL e faz proxy do Piper TTS,
que roda no mesmo container escutando só em `127.0.0.1`.

> O diretório ainda se chama `webgl` por causa do vínculo já existente com o serviço no
> Railway. Ele serve as duas coisas.

## Por que tudo junto

Não é para economizar um serviço — é para eliminar CORS.

| | Serviços separados | Tudo junto |
|---|---|---|
| CORS no TTS | precisa configurar `ALLOWED_ORIGINS` | não existe: mesma origem |
| Domínio no código Unity | hardcoded, quebra ao mudar | derivado de `Application.absoluteURL` |
| Deploy | dois | um |

O mesmo raciocínio que resolveu os personagens (§8.4 do MD3): o que está na mesma origem
não passa por CORS.

**O custo:** a imagem fica grande (python + onnxruntime + espeak + vozes, além dos 56 MB
do site), o ciclo de vida acopla — rebuild do WebGL redeploya o TTS — e não dá para
escalar o TTS separado.

## Rotas

| Caminho | Destino |
|---|---|
| `/` | build WebGL (`/srv`) |
| `/entrar` | sorteio no RAG; 302 para `/e/1` ou `/e/2` — **é este que se distribui** |
| `/e/1`, `/e/2` | o mesmo build, por rewrite interno (grupo masculino / feminino) |
| `/homem`, `/mulher` | 302 para `/e/1` e `/e/2` — atalhos do facilitador |
| `/Build/*.gz` | assets com `Content-Encoding: gzip` |
| `/tts/*` | Piper interno (`127.0.0.1:9002`); o prefixo é removido |
| `/rag/*` | serviço do RAG na rede privada; carrega `X-Sessao`/`X-Grupo` |
| `/ride-content-e22e17c3/*` | proxy do bucket de personagens da USC |

## Backend de TTS

Dois backends atendem o **mesmo contrato** (`/health`, `/voices`, `/synthesize`), que é o que o
`TextToSpeechSystemPiper` do RIDE consome. Trocar entre eles é variável de ambiente — **não
exige rebuild do Unity**.

| `TTS_BACKEND` | Serviço | Voz pt-BR |
|---|---|---|
| `deepinfra` | Kokoro-82M na DeepInfra ([tts_kokoro/main.py](tts_kokoro/main.py)) | `pf_dora` (feminina) |
| `piper` | Piper local ([app/main.py](app/main.py), submódulo RideServices) | `pt_BR-faber-medium` (masculina) |

Sem `TTS_BACKEND` definido, a presença de `DEEPINFRA_TOKEN` decide: com token vai de Kokoro,
sem token cai no Piper — um deploy sem chave não fica mudo.

### Voz por personagem

O `TryApplyCharacterVoice` do RIDE resolve a voz **por nome**
(`GetVoiceIndex(profile.PiperVoiceName)`), não por índice cego — por isso expor várias vozes é
seguro, desde que os nomes batam. Os 38 perfis do `Sample_Characters_Prefab.prefab` foram
mapeados por gênero:

| Personagem | Voz |
|---|---|
| `Ellie` e demais `*_Female_*`, `ccMaria`, `ccMila`, `ccAriana` (12) | `pf_dora` |
| `KevinCivilian`, `KevinArmy` (2) | `pm_santa` |
| demais masculinos (24) | `pm_alex` |

Um nome que não conste em `/voices` cai em `voices[0]`, então **a primeira da lista é o
fallback**. Mudar o mapa exige rebuild do Unity (é dado de prefab) — mas trocar o *timbre* não:

```
KOKORO_VOICE_MAP=pm_santa=pm_alex    # tira o tom natalino do Kevin, sem rebuild
```

`GET /tts/health` mostra as vozes ativas e o mapa em vigor.

**Por que adaptar o Piper em vez de usar o `TextToSpeechSystemKokoro` do RIDE:** aquele cliente
grava o áudio em arquivo e devolve um caminho, o que não carrega em WebGL, e tem o endpoint
privado e hardcoded. Usá-lo custaria dois patches no pacote mais um rebuild. O cliente Piper já
está patchado para data URI e já deriva `{origin}/tts`, então trocar o que está atrás dele sai
de graça. **O custo é uma mentira de rótulo:** o menu de debug continua dizendo "Piper (Local)".

A chave da DeepInfra fica só no servidor. O navegador fala com `/tts` na mesma origem, sem
chave e sem CORS — pela mesma razão de sempre: qualquer segredo dentro de um build WebGL é
público.

Para testar local sem expor o token no shell:

```bash
echo 'DEEPINFRA_TOKEN=...' > deploy/webgl/.env.deepinfra   # já no .gitignore/.dockerignore
docker run --rm --env-file deploy/webgl/.env.deepinfra -p 8080:8080 vhtoolkit-webgl
```

## Estudo: links de entrada e código de sessão

Distribua **um único link a todo mundo**: `/entrar`. Quem decide o grupo é o servidor.

| Link | Para quem | O que faz |
|---|---|---|
| `/entrar` | **participantes** | sorteia o grupo e redireciona para `/e/1` ou `/e/2` |
| `/e/1`, `/e/2` | — | as duas condições; é o que fica na barra de endereços |
| `/homem`, `/mulher` | facilitador | atalhos memoráveis; redirecionam para `/e/1` e `/e/2` |

> **Não mande `/homem` nem `/mulher` a participante nenhum.** O nome entrega o grupo antes
> mesmo de a página abrir, e um participante que sabe em que condição está deixa de ser
> cego ao experimento.

Os caminhos são opacos de propósito. A barra de endereços fica visível a conversa inteira, e
um `?genero=feminino` ali contaria ao participante em que condição ele está. Com `/e/1` e
`/e/2` dá para perceber que existem dois grupos, mas não qual é o seu.

`/e/1` e `/e/2` são servidos por **rewrite interno**, não redirecionamento: o navegador
continua mostrando o caminho enquanto o servidor entrega o `index.html`. Isso importa porque
o Unity lê o grupo de `Application.absoluteURL`
(`DemoControllerBase.ResolveForcedCharacterName`), e assim a URL já nasce certa — nada
precisa ser alterado depois da carga, sem corrida com o `Start()` da cena.

### O sorteio equilibrado

Sortear cada visitante de forma independente não garante grupos iguais: com 100
participantes, a chance de terminar com desvio de 6 ou mais passa de 27%, e grupos
desbalanceados custam poder estatístico justamente no teste que o estudo quer fazer.

Por isso `/entrar` usa **blocos permutados** (`src/api/alocacao.py`, no serviço do RAG): a
ordem é sorteada dentro de blocos que já contêm metade de cada grupo. A cada 4 participantes
fecham exatamente 2 e 2; ao fim de 100, 50 e 50 exatos — e a ordem dentro do bloco continua
imprevisível. O tamanho do bloco sai de `ESTUDO_TAMANHO_BLOCO` (padrão 4, precisa ser par).

O sorteio acontece no **servidor**, antes de a página carregar, por duas razões: o Unity já
encontra o caminho certo quando a cena inicia, e o contador do equilíbrio vive num lugar só —
espalhado pelos navegadores, não haveria como equilibrar. O estado fica num arquivo no
volume, então sobrevive a redeploys.

```bash
curl -H "Authorization: Bearer $LOGS_TOKEN" https://SEU-DOMINIO/rag/estudo/estado
curl -X POST -H "Authorization: Bearer $LOGS_TOKEN" https://SEU-DOMINIO/rag/estudo/reiniciar
```

O `reiniciar` zera só a contagem do sorteio — nenhuma conversa é apagada. Use depois do
ensaio, para que a contagem valha só para os participantes reais.

### O código de sessão

`inject-session.py` (aplicado pelo `sync-build.sh`, como o `inject-diagnostics.py`) põe na
tela um badge com um código curto — `VH-7K2M-4QX9`, base32 de Crockford, sem `I`, `L`, `O`
nem `U` para não confundir quem digita. O participante copia esse código e cola no campo
correspondente do questionário; é o **único** elo entre a resposta do formulário e a
conversa registrada no servidor. Não há login, não há nome: o código é aleatório e não tem
nada do participante dentro.

O mesmo código vai para um cookie de mesma origem, e é assim que o servidor fica sabendo
dele **sem rebuild do Unity**: o navegador anexa o cookie sozinho às chamadas de `/rag`
(mesma origem, de novo — o mesmo raciocínio que resolveu o CORS do TTS e dos personagens),
o Caddy traduz cookie em header (`X-Sessao`, `X-Grupo`) e o serviço do RAG grava a conversa
sob esse código.

| Momento | Código |
|---|---|
| Abre o link | novo, sempre |
| F5 no meio da conversa | o mesmo |
| Fecha a aba | morre junto (fica em `sessionStorage`) |

A distinção entre "abriu o link" e "deu F5" vem do navegador, não da URL: a API de navigation
timing diz se a navegação foi `reload`. Antes isso exigia um parâmetro na URL que precisava
ser apagado em seguida; agora a URL fica intocada.

Para o badge também abrir o questionário, defina `URL_FORMULARIO` antes do sync, com `{ID}`
onde entra o código:

```bash
URL_FORMULARIO='https://docs.google.com/forms/d/e/XXXX/viewform?usp=pp_url&entry.123456={ID}' \
  ./deploy/webgl/sync-build.sh
```

Sem a variável, o badge mostra só o código e o botão de copiar.

### Onde as conversas ficam

No volume do serviço do RAG, um arquivo JSONL por sessão (`LOG_DIR`, padrão `/data/logs`),
uma linha por turno: pergunta, resposta, rota (conversa / fundamentada / sem contexto /
erro), ferramentas acionadas, distâncias das passagens e durações. Erro da LLM também vira
linha — o que não pode acontecer é o turno sumir.

Para ler, defina `LOGS_TOKEN` no serviço do RAG (sem ele, os endpoints respondem 403 de
propósito: `/rag/*` é público através daqui):

```bash
curl -H "Authorization: Bearer $LOGS_TOKEN" https://SEU-DOMINIO/rag/logs/sessoes
curl -H "Authorization: Bearer $LOGS_TOKEN" https://SEU-DOMINIO/rag/logs/sessoes/VH-7K2M-4QX9
curl -H "Authorization: Bearer $LOGS_TOKEN" -O https://SEU-DOMINIO/rag/logs/export.csv
```

O `export.csv` tem uma linha por mensagem e a coluna `sessao` — é por ela que se cruza com
a planilha de respostas do formulário.

## Sem NLP não há voz

Em WebGL toda fala nasce de uma resposta do NLP: o personagem sintetiza o texto que o NLP
devolveu, nunca o que foi digitado. Se o NLP não responde, não há texto, o TTS nunca é
chamado e o personagem fica mudo — enquanto `/tts/voices` segue respondendo normalmente. É
por isso que `/rag` importa tanto quanto `/tts` para a voz funcionar.

O `NlpSystemVLLM.SystemInit` deriva o endpoint de `Application.absoluteURL` e aponta para
`{origem}/rag/v1/chat/completions`: mesma origem, sem CORS a configurar, servido pelo RAG de
auditoria parlamentar (ver Caddyfile). O personagem fala a resposta do RAG.

> **Nota histórica.** Antes do RAG próprio subir, o endpoint caía no API Gateway da USC, que
> só libera CORS para o domínio deles — a requisição morria no preflight e o personagem nunca
> falava. Um `inject-voice-shim.py` interceptava aquele `fetch` e devolvia o próprio texto
> digitado, sem consultar LLM nenhum, só para validar VHToolkit + TTS ponta a ponta. O shim
> foi removido junto com a chamada no `sync-build.sh`; referência a ele ou a `VHLAB_VOZ` em
> qualquer lugar é resíduo.

## Passos

```bash
./deploy/webgl/sync-build.sh     # build WebGL + instrumentação + código do Piper
docker build -t vhtoolkit-webgl ./deploy/webgl
docker run --rm -p 8080:8080 vhtoolkit-webgl
```

Validação local:

```bash
curl -sI http://localhost:8080/Build/WebGL.wasm.gz | grep -i "content-encoding"
curl -s  http://localhost:8080/tts/voices
curl -s -X POST http://localhost:8080/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Teste de voz","voice":"pt_BR-faber-medium"}' | head -c 120
```

Subir:

```bash
cd deploy/webgl && railway up --no-gitignore
```

> **`--no-gitignore` é obrigatório.** `site/`, `app/` e `requirements.txt` estão no
> `.gitignore` (são artefatos copiados, não fonte). Sem a flag o contexto sobe quase vazio
> e o build falha no `COPY` com `not found`.

## Variáveis no Railway

Hoje **nenhuma** está definida no painel: os valores vêm do `ENV` do Dockerfile, o que mantém
a imagem idêntica em local e em produção. Definir no Railway sobrescreve — em particular,
`SUPPORTED_VOICES` com mais de uma voz reabre o problema descrito em "Vozes".

```
DEEPINFRA_TOKEN=<a chave>          # define-se pelo painel; habilita o backend Kokoro
TTS_BACKEND=deepinfra              # opcional: sem isso, a presença do token decide
KOKORO_DEFAULT_VOICE=pf_dora
KOKORO_VOICES=pf_dora
DEFAULT_VOICE=pt_BR-faber-medium   # usados só quando TTS_BACKEND=piper
SUPPORTED_VOICES=pt_BR-faber-medium
API_TOKEN=
```

`ALLOWED_ORIGINS` não é mais necessária — não há requisição cross-origin. `API_TOKEN` fica
vazio de propósito: qualquer token embutido num build WebGL é público e não protegeria nada.

## Vozes

Ficam embutidas na imagem em build time. Baixar a voz no primeiro request levou **9,8s**
na medição local, contra **0,8s** já aquecido — e o cliente Unity corta em 20s. Ao mudar
`SUPPORTED_VOICES`, ajuste também as linhas `piper.download_voices` do Dockerfile.

**Só `pt_BR-faber-medium` é exposta, e isso é deliberado.** O Unity escolhe a voz por
índice na lista que `GET /voices` devolve (`DemoControllerBase.CreateTTS` passa
`voices[m_ttsVoice]`), e o `ResolveVoiceOrDefault` do Piper só cai no padrão pt-BR quando a
voz pedida é inválida. Com duas vozes na lista, um índice 1 faz o personagem ler texto em
português com voz em inglês. Com uma só, não há escolha errada possível.

## Licença

`piper-tts==1.4.2` é **GPL-3.0-or-later** (linha OHF-Voice/piper1-gpl); o último release
MIT, o 1.2.0, não tem os módulos que o serviço usa. Rodar como serviço de rede não dispara
as obrigações de distribuição da GPL — isso seria AGPL. Se um dia distribuírem binários que
embutam essa dependência, a questão volta. O repositório do RIDE trata isso como decisão
em aberto.
