# Imagem única: serve o build WebGL e roda o TTS na mesma origem.
#
# Por que junto, e não em dois serviços: na mesma origem o navegador nunca faz requisição
# cross-origin para o TTS, então não há CORS a configurar, e o build não precisa saber o
# domínio de produção — ele deriva do endereço da própria página.
#
# O Caddy entra por multi-stage: é binário Go estático, roda no Debian.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TTS_PORT=9002

COPY --from=caddy:2-alpine /usr/bin/caddy /usr/bin/caddy

WORKDIR /app

COPY requirements-kokoro.txt ./
RUN pip install --no-cache-dir -r requirements-kokoro.txt

# TTS: Kokoro-82M na DeepInfra, servido no MESMO contrato que o cliente Piper do RIDE
# consome (/health, /voices, /synthesize) — por isso o build WebGL não muda.
#
# O Piper local saiu: só tinha vozes masculinas em pt-BR, e arrastava espeak-ng, o
# onnxruntime e o download da voz ONNX em build time. De quebra, resolve a pendência de
# licença: o piper-tts 1.3.0+ é GPL-3.0-or-later.
COPY tts_kokoro ./tts_kokoro
COPY Caddyfile /etc/caddy/Caddyfile
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY site /srv

RUN chmod +x /usr/local/bin/entrypoint.sh

# Config default do TTS. Fica aqui, e nao so no painel do Railway, para a imagem
# funcionar identica em local. Variaveis de ambiente do Railway sobrescrevem, se precisar.
#
# As tres vozes pt-BR ficam expostas porque o RIDE resolve voz por NOME, e cada personagem
# no Sample_Characters_Prefab aponta para uma delas (Ellie -> pf_dora, Kevin -> pm_santa,
# demais por genero). Nome que nao casar com nenhuma sai na KOKORO_DEFAULT_VOICE — inclusive
# "pt_BR-faber-medium", que continua compilado no build como fallback do cliente.
# KOKORO_VOICE_MAP troca o timbre sem rebuild: "pm_santa=pm_alex" tira o tom natalino do
# Kevin mantendo o prefab intacto.
ENV KOKORO_DEFAULT_VOICE=pf_dora \
    KOKORO_VOICES=pf_dora,pm_alex,pm_santa \
    KOKORO_VOICE_MAP=""

EXPOSE 8080
CMD ["/usr/local/bin/entrypoint.sh"]
