#!/usr/bin/env bash
# Copia o build WebGL mais recente para ./site, que e o que vai para a imagem.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$RAIZ/vhtoolkit-vhlab/VHUnityURP/Build/WebGL"
DEST="$(cd "$(dirname "$0")" && pwd)/site"

if [ ! -f "$SRC/index.html" ]; then
  echo "erro: build nao encontrado em $SRC" >&2
  echo "rode o build antes (ver deploy/webgl/README.md)" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$SRC"/. "$DEST"/

echo "build copiado para $DEST"

# O index.html e regerado a cada build do Unity, entao a instrumentacao e reaplicada aqui.
AQUI="$(cd "$(dirname "$0")" && pwd)"
python3 "$AQUI/inject-diagnostics.py" "$DEST/index.html"

# Codigo de sessao do estudo: o badge que o participante copia para o questionario e os
# cookies que o Caddy traduz em X-Sessao/X-Grupo para o RAG registrar a conversa.
# Defina URL_FORMULARIO (com {ID} onde entra o codigo) para o badge abrir o questionario.
python3 "$AQUI/inject-session.py" "$DEST/index.html"

# O shim de voz foi removido: o NLP agora responde de verdade, pelo RAG de auditoria
# parlamentar servido em /rag (ver Caddyfile e NlpSystemVLLM.SystemInit). O personagem
# fala a resposta do RAG, nao o texto digitado.

# O TTS nao e mais sincronizado daqui: o Piper local saiu do deploy e o unico backend e o
# Kokoro-82M na DeepInfra, cujo adaptador vive em ./tts_kokoro e nao vem de submodulo.

du -sh "$DEST"
