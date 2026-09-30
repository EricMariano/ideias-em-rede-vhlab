#!/usr/bin/env bash
# Sobe o TTS (interno) e o Caddy (exposto). Se qualquer um dos dois morrer, o container
# encerra para o Railway reiniciar — em vez de ficar de pé servindo meio serviço.
set -euo pipefail

# TTS: Kokoro-82M na DeepInfra, unico backend. O contrato servido (/health, /voices,
# /synthesize) continua sendo o que o TextToSpeechSystemPiper do RIDE consome, entao o
# build WebGL nao muda. O Piper local saiu: so tinha vozes masculinas em pt-BR, arrastava
# espeak-ng e onnxruntime para a imagem, e o piper-tts e GPL-3.0.
#
# Sem token nao ha como falar, e um container de pe servindo TTS mudo e pior que um que
# nao sobe: falha aqui, alto e claro, para o Railway mostrar o motivo no deploy.
if [ -z "${DEEPINFRA_TOKEN:-}" ]; then
  echo "[entrypoint] DEEPINFRA_TOKEN nao configurado — o TTS nao tem como sintetizar." >&2
  echo "[entrypoint] Defina a variavel no servico do Railway e refaca o deploy." >&2
  exit 1
fi

echo "[entrypoint] TTS backend=deepinfra (tts_kokoro.main:app)"

uvicorn tts_kokoro.main:app --host 127.0.0.1 --port "${TTS_PORT:-9002}" &
pid_tts=$!

caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
pid_web=$!

# Encerra o irmão sobrevivente antes de sair, para nao deixar processo orfao.
trap 'kill "$pid_tts" "$pid_web" 2>/dev/null || true' EXIT

wait -n "$pid_tts" "$pid_web"
codigo=$?
echo "[entrypoint] um dos processos encerrou (código $codigo); derrubando o container"
exit "$codigo"
