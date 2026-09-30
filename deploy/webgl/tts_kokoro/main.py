"""Adaptador Kokoro (DeepInfra) servido no contrato do serviço Piper.

Por que existe: o `TextToSpeechSystemPiper` do RIDE já está patchado para WebGL (devolve
data URI em vez de caminho de arquivo) e já aponta para `{origin}/tts`. O
`TextToSpeechSystemKokoro`, não: ele grava arquivo em disco — inútil em WebGL — e tem
endpoint privado e hardcoded, o que exigiria rebuild do Unity.

Então, em vez de trocar o cliente, trocamos o backend. Este módulo expõe exatamente as três
rotas que o cliente Piper consome (`/health`, `/voices`, `/synthesize`), com os mesmos nomes
de campo, e por baixo chama o Kokoro-82M na DeepInfra. O build WebGL não muda em nada.

Este é o único backend de TTS do deploy: o Piper local foi removido (ver Dockerfile). O
contrato do cliente Piper continua sendo o que falamos, porque é o que está compilado no
build WebGL — o rótulo "Piper (Local)" no menu de debug é mentira de rótulo, consciente.
Corrigir o rótulo exige rebuild; quando vier o rebuild do RAG, vale renomear.

A chave nunca vai para o navegador: ela vive aqui, no servidor, e o navegador fala com
`/tts` na mesma origem da página.
"""

import base64
import io
import logging
import os
import wave
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


API_TOKEN = os.getenv("API_TOKEN", "").strip()
DEEPINFRA_TOKEN = os.getenv("DEEPINFRA_TOKEN", "").strip()
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "hexgrad/Kokoro-82M").strip()
DEEPINFRA_URL = os.getenv(
    "DEEPINFRA_URL", f"https://api.deepinfra.com/v1/inference/{DEEPINFRA_MODEL}"
).strip()

# Expor varias vozes e seguro: o TryApplyCharacterVoice do RIDE resolve por NOME
# (GetVoiceIndex(profile.PiperVoiceName)), nao por indice cego. Cada personagem no
# Sample_Characters_Prefab aponta para um destes nomes. Um nome que nao esteja na lista
# cai em voices[0], entao a ordem importa: a primeira e o fallback.
DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "pf_dora").strip()
SUPPORTED_VOICES = [
    voice.strip()
    for voice in os.getenv("KOKORO_VOICES", "pf_dora,pm_alex,pm_santa").split(",")
    if voice.strip()
]


def _parse_voice_map(raw: str) -> dict[str, str]:
    """Traduz nomes anunciados em /voices para o preset real da DeepInfra.

    Existe para desacoplar o que esta gravado no prefab (e so muda com rebuild do Unity)
    do timbre que sai de fato (que deve poder mudar com uma variavel de ambiente).
    Ex.: KOKORO_VOICE_MAP="pm_santa=pm_alex" troca a voz do Kevin sem rebuildar nada.
    """
    mapa: dict[str, str] = {}
    for par in raw.split(","):
        nome, _, destino = par.partition("=")
        if nome.strip() and destino.strip():
            mapa[nome.strip()] = destino.strip()
    return mapa


VOICE_MAP = _parse_voice_map(os.getenv("KOKORO_VOICE_MAP", ""))

# O cliente Unity corta em 20s (TextToSpeechSystemPiper.m_requestTimeoutSeconds), entao o
# nosso teto fica abaixo disso para o erro chegar como resposta, e nao como timeout mudo.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("DEEPINFRA_TIMEOUT", "15"))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]


logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="VHToolkit Kokoro TTS (DeepInfra)")

# Mesma origem no deploy (o Caddy publica isto sob /tts), entao CORS aqui e so para uso
# direto em teste local.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    default_voice: str
    backend: str
    token_configured: bool
    voices: list[str]
    voice_map: dict[str, str]


class VoicesResponse(BaseModel):
    voices: list[str]


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class SynthesizeResponse(BaseModel):
    audio_base64: str
    audio_format: str
    sample_rate_hz: int
    duration_seconds: float
    voice: str


def require_authorization(authorization: Optional[str]) -> None:
    if not API_TOKEN:
        return

    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def resolve_voice(requested_voice: Optional[str]) -> str:
    """Resolve o nome pedido para uma voz real, caindo na padrao em vez de recusar.

    Recusar com 400 deixava o humano virtual MUDO: o cliente Unity trata qualquer erro de
    sintese como "sem audio" e segue em frente, so com um Debug.LogWarning no console. E o
    fallback dele (TextToSpeechSystemPiper.m_fallbackVoice) e "pt_BR-faber-medium", nome do
    Piper que nao existe mais aqui — entao bastava um GET /voices falhar uma vez para o
    build travar nesse nome e nunca mais falar. Nome desconhecido agora sai na voz padrao:
    timbre errado e melhor que silencio, e nao exige rebuild do Unity.
    """
    if requested_voice and requested_voice.strip():
        normalized_voice = requested_voice.strip()
        if normalized_voice in SUPPORTED_VOICES:
            return normalized_voice
        logger.warning(
            "Voz '%s' desconhecida; usando a padrao '%s'.", normalized_voice, DEFAULT_VOICE
        )

    return DEFAULT_VOICE


def extract_audio_bytes(response: httpx.Response) -> tuple[bytes, float]:
    """Devolve (bytes do áudio, duração conhecida ou 0.0).

    A spec pública da DeepInfra deixa o schema de resposta vazio, e as duas formas
    aparecem na documentação deles: JSON com o campo `audio` (data URI ou base64 puro) e
    corpo binário direto. Aceitamos as três em vez de apostar em uma.
    """
    content_type = response.headers.get("content-type", "")

    if "application/json" not in content_type:
        return response.content, 0.0

    payload = response.json()
    audio = payload.get("audio")
    if not audio:
        detail = payload.get("detail") or payload.get("error") or "resposta sem campo 'audio'"
        raise HTTPException(status_code=502, detail=f"DeepInfra: {detail}")

    if audio.startswith("data:"):
        _, _, audio = audio.partition(",")

    try:
        audio_bytes = base64.b64decode(audio)
    except Exception as exception:  # noqa: BLE001 - qualquer falha aqui e payload invalido
        raise HTTPException(status_code=502, detail=f"DeepInfra: base64 invalido ({exception}).") from exception

    # A DeepInfra devolve timings por palavra; o fim da ultima serve de duracao quando o
    # WAV nao puder ser lido. O lipsync do RIDE depende dessa duracao.
    duracao = 0.0
    words = payload.get("words") or []
    if words:
        try:
            duracao = float(words[-1].get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            duracao = 0.0

    return audio_bytes, duracao


def get_wav_metadata(audio_bytes: bytes) -> tuple[int, float]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration_seconds = frame_count / float(sample_rate) if sample_rate > 0 else 0.0
        return sample_rate, duration_seconds


def synthesize_with_deepinfra(text: str, voice: str) -> tuple[bytes, float]:
    if not DEEPINFRA_TOKEN:
        raise HTTPException(status_code=503, detail="DEEPINFRA_TOKEN nao configurado.")

    try:
        response = httpx.post(
            DEEPINFRA_URL,
            headers={
                "Authorization": f"Bearer {DEEPINFRA_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "preset_voice": VOICE_MAP.get(voice, voice),
                # wav para conseguirmos ler sample rate e duracao reais do cabecalho.
                "output_format": "wav",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exception:
        raise HTTPException(status_code=502, detail=f"DeepInfra inalcancavel: {exception}") from exception

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"DeepInfra respondeu {response.status_code}: {response.text[:300]}",
        )

    return extract_audio_bytes(response)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        default_voice=DEFAULT_VOICE,
        backend=f"deepinfra:{DEEPINFRA_MODEL}",
        token_configured=bool(DEEPINFRA_TOKEN),
        voices=SUPPORTED_VOICES,
        voice_map=VOICE_MAP,
    )


@app.get("/voices", response_model=VoicesResponse)
def voices() -> VoicesResponse:
    return VoicesResponse(voices=SUPPORTED_VOICES)


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(
    request: SynthesizeRequest,
    authorization: Optional[str] = Header(default=None),
) -> SynthesizeResponse:
    require_authorization(authorization)

    text = request.text.strip() if request.text else ""
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    voice = resolve_voice(request.voice)
    audio_bytes, duracao_informada = synthesize_with_deepinfra(text, voice)

    if not audio_bytes:
        raise HTTPException(status_code=502, detail="DeepInfra devolveu audio vazio.")

    try:
        sample_rate_hz, duration_seconds = get_wav_metadata(audio_bytes)
    except wave.Error:
        # Nao veio WAV: ainda entregamos o audio, mas a duracao tem que sair do que a
        # DeepInfra informou, senao o lipsync fica sem escala.
        sample_rate_hz, duration_seconds = 0, duracao_informada

    if duration_seconds <= 0.0:
        duration_seconds = duracao_informada

    return SynthesizeResponse(
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
        audio_format="wav",
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
        voice=voice,
    )
