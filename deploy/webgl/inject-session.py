#!/usr/bin/env python3
"""Injeta o identificador de sessao do estudo no index.html gerado pelo Unity.

Cada participante entra por /e/1 ou /e/2 (ver Caddyfile), recebe um codigo curto
e o transcreve no formulario. Os caminhos sao opacos de proposito: a barra de enderecos
fica a vista a conversa inteira, e um "?genero=feminino" ali entregaria a condicao do
participante. Esse codigo e a unica coisa que liga a resposta do
questionario a conversa registrada no servidor do RAG — nao ha login, nao ha nome, e o
que sai daqui e um identificador aleatorio sem nada do participante dentro.

O codigo vive em tres lugares, de proposito:

  - na tela, num badge que fica visivel a conversa inteira (e o que se copia);
  - no sessionStorage, para sobreviver a um F5 acidental e morrer ao fechar a aba;
  - num cookie, que o navegador anexa sozinho as chamadas de /rag por serem de mesma
    origem. E assim que o servidor descobre a sessao SEM rebuild do WebGL: o Caddy
    traduz o cookie em header (X-Sessao) e o RAG registra a conversa sob esse codigo.

O index.html e regerado a cada build do Unity, entao a injecao roda no sync-build.sh,
depois da copia — mesmo racional do inject-diagnostics.py.

Uso:
    python3 inject-session.py site/index.html

Opcional, para o badge tambem abrir o questionario ja com o codigo preenchido:
    URL_FORMULARIO='https://forms.gle/xxx?entry.123={ID}' python3 inject-session.py ...

O trecho `{ID}` e substituido pelo codigo da sessao. Sem a variavel, o badge mostra so o
codigo e o botao de copiar.
"""
import json
import os
import pathlib
import sys

MARCA = "<!-- vhlab-sessao -->"

# O index.html do Unity referencia os assets por caminho RELATIVO ("Build/...",
# "TemplateData/..."). Servido na raiz isso funciona, mas o estudo serve a MESMA pagina em
# /e/1 e /e/2 (rewrite interno no Caddy, para o grupo ficar na URL sem aparecer), e dali o
# navegador resolveria "Build/x.js" como "/e/Build/x.js" -> 404, tela em branco.
#
# <base href="/"> ancora toda resolucao relativa na raiz, seja qual for o caminho em que a
# pagina foi servida. Precisa vir no <head>, antes de qualquer referencia.
#
# Nao afeta a deteccao do grupo: ela le document.URL, que continua sendo /e/1 ou /e/2. Nem
# os endpoints de /tts e /rag, que o RIDE monta a partir da ORIGEM (GetPageOrigin corta no
# primeiro "/" depois do host), nao do caminho.
MARCA_BASE = "<!-- vhlab-base -->"

# O template do Unity dispara o download (41 MB) assim que a pagina abre. No estudo isso e
# cedo demais: o participante cai direto numa barra de progresso, sem saber o que vai
# acontecer, e quem abre o link e desiste leva os 41 MB junto. Trocamos a linha que anexa o
# loader por um gancho, que a tela de inicio chama no clique em "Comecar".
ANCORA_BOOT = "document.body.appendChild(script);"
GANCHO_BOOT = (
    "window.__vhlabIniciarUnity = function () { document.body.appendChild(script); };"
)

# A barra de progresso do proprio Unity so deve aparecer depois do clique; antes dele a tela
# de inicio cobre tudo, e uma barra parada por baixo so confunde.
ANCORA_BARRA = 'document.querySelector("#unity-loading-bar").style.display = "block";'
BASE = '  <base href="/">\n  ' + MARCA_BASE + "\n"

MODELO = """<!-- vhlab-sessao -->
<style>
  #vhlab-sessao {
    position: fixed; top: 10px; right: 10px; z-index: 99998;
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 10px;
    background: rgba(17, 17, 17, .82); color: #f2f2f2;
    border: 1px solid rgba(255, 255, 255, .18);
    font: 12px/1.35 -apple-system, system-ui, "Segoe UI", sans-serif;
    box-shadow: 0 2px 12px rgba(0, 0, 0, .35);
    opacity: .85; transition: opacity .15s;
    -webkit-user-select: none; user-select: none;
  }
  #vhlab-sessao:hover { opacity: 1; }
  #vhlab-sessao .rotulo {
    text-transform: uppercase; letter-spacing: .06em;
    font-size: 10px; color: #b9b9b9;
  }
  #vhlab-sessao .codigo {
    font: 600 15px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: .06em; color: #fff;
    -webkit-user-select: all; user-select: all;
  }
  #vhlab-sessao button {
    cursor: pointer; border-radius: 7px; padding: 6px 10px;
    font: 600 12px/1 -apple-system, system-ui, sans-serif;
    border: 1px solid rgba(255, 255, 255, .28);
    background: rgba(255, 255, 255, .1); color: #fff;
  }
  #vhlab-sessao button:active { transform: translateY(1px); }
  #vhlab-sessao button.ok { background: #1f7a43; border-color: #1f7a43; }
  #vhlab-sessao button.destaque { background: #1f5f9e; border-color: #2a72b8; }
  #vhlab-sessao button:disabled { opacity: .4; cursor: default; }
  #vhlab-sessao .estado {
    font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: #8fbf9f;
  }
  #vhlab-sessao .estado.encerrada { color: #d8a657; }
  @media (max-width: 600px) {
    #vhlab-sessao { top: 6px; right: 6px; left: 6px; padding: 6px 10px; }
  }

  #vhlab-inicio {
    position: fixed; inset: 0; z-index: 99997;
    display: flex; align-items: center; justify-content: center;
    background: #15171a; color: #eceff1;
    font: 15px/1.6 -apple-system, system-ui, "Segoe UI", sans-serif;
    padding: 24px; overflow: auto;
  }
  #vhlab-inicio .painel { max-width: 460px; }
  #vhlab-inicio h1 { font-size: 21px; margin: 0 0 16px; font-weight: 600; }
  #vhlab-inicio ol { margin: 0 0 22px; padding-left: 20px; color: #c2c8cc; }
  #vhlab-inicio li { margin-bottom: 9px; }
  #vhlab-inicio .aviso { color: #98a0a6; font-size: 13px; margin: 0 0 22px; }
  #vhlab-inicio button {
    cursor: pointer; width: 100%; padding: 14px 20px; border-radius: 9px;
    font: 600 16px/1 -apple-system, system-ui, sans-serif;
    border: 0; background: #2a72b8; color: #fff;
  }
  #vhlab-inicio button:active { transform: translateY(1px); }
  #vhlab-inicio button:disabled { background: #3c4145; cursor: default; }
</style>
<script>
(function () {
  'use strict';

  var CHAVE = 'vhlab_sessao';
  // Crockford base32: sem I, L, O e U. Quem digitar o codigo a mao no formulario nao tem
  // como confundir 1 com I nem 0 com O, e nao ha como o sorteio formar palavra.
  var ALFABETO = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
  var URL_FORMULARIO = __URL_FORMULARIO__;

  function sortear(tamanho) {
    var saida = '';
    var bytes = new Uint8Array(tamanho);
    if (window.crypto && window.crypto.getRandomValues) {
      window.crypto.getRandomValues(bytes);
    } else {
      for (var j = 0; j < tamanho; j++) bytes[j] = Math.floor(Math.random() * 256);
    }
    for (var i = 0; i < tamanho; i++) saida += ALFABETO[bytes[i] % ALFABETO.length];
    return saida;
  }

  // 8 caracteres em base32 = 32 bits de entropia. Para algumas centenas de sessoes a
  // chance de colisao e desprezivel, e o codigo continua curto para transcrever.
  function novoCodigo() { return 'VH-' + sortear(4) + '-' + sortear(4); }

  function ler() { try { return sessionStorage.getItem(CHAVE); } catch (_) { return null; } }
  function guardar(v) { try { sessionStorage.setItem(CHAVE, v); } catch (_) {} }

  function gravarCookie(nome, valor) {
    // 12 h cobre com folga qualquer sessao de coleta e evita que um cookie esquecido
    // num equipamento compartilhado carimbe a conversa do participante seguinte.
    try {
      document.cookie = nome + '=' + encodeURIComponent(valor) +
        ';path=/;max-age=43200;samesite=Lax';
    } catch (_) {}
  }

  // O grupo vem do CAMINHO (/e/1, /e/2), nao da query string: a barra de enderecos fica
  // visivel ao participante a conversa inteira, e "?genero=feminino" ali contaria a ele em
  // que condicao esta. Mesma leitura que o DemoControllerBase.ResolveForcedCharacterName
  // faz do lado do Unity — os dois precisam concordar.
  var caminho = window.location.pathname;
  while (caminho.length > 1 && caminho.charAt(caminho.length - 1) === '/')
    caminho = caminho.slice(0, -1);
  var grupo = 'indefinido';
  if (caminho.slice(-4) === '/e/1') grupo = 'masculino';
  else if (caminho.slice(-4) === '/e/2') grupo = 'feminino';

  if (grupo === 'indefinido') {
    // Atalho de teste manual, aceito tambem pelo lado do Unity.
    var genero = (new URLSearchParams(window.location.search).get('genero') || '')
      .trim().toLowerCase();
    if (genero === 'feminino' || genero === 'female' || genero === 'f') grupo = 'feminino';
    else if (genero === 'masculino' || genero === 'male' || genero === 'm') grupo = 'masculino';
  }

  // Participante novo ou F5 do mesmo participante? Antes isso vinha de um `novo=1` na URL,
  // que precisava ser apagado em seguida; agora o navegador ja sabe responder. Recarga e
  // voltar/avancar preservam o codigo ja anotado; qualquer outra entrada cunha um novo,
  // inclusive abrir o mesmo link de novo na mesma aba para o proximo participante.
  function ehRecarga() {
    try {
      var entradas = performance.getEntriesByType('navigation');
      if (entradas && entradas.length)
        return entradas[0].type === 'reload' || entradas[0].type === 'back_forward';
      // Navegador antigo: a API depreciada responde 1 para reload, 2 para back/forward.
      if (performance.navigation) return performance.navigation.type !== 0;
    } catch (_) {}
    // Sem como saber, preserva: um codigo repetido se conserta na analise, um codigo
    // perdido no meio da conversa nao.
    return true;
  }

  var codigo = ehRecarga() ? ler() : null;
  if (!codigo) { codigo = novoCodigo(); guardar(codigo); }

  gravarCookie('vhlab_sid', codigo);
  gravarCookie('vhlab_grupo', grupo);

  window.VHLAB_SESSAO = {
    codigo: codigo,
    grupo: grupo,
    comecar: comecar,
    finalizar: function () { finalizar(null); },
    estado: function () { return estado; },
  };

  function copiar(botao) {
    var original = botao.textContent;
    function confirmou() {
      botao.textContent = 'copiado';
      botao.className = 'ok';
      setTimeout(function () { botao.textContent = original; botao.className = ''; }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(codigo).then(confirmou, function () { selecionar(); });
      return;
    }
    selecionar();
    function selecionar() {
      // Sem permissao de clipboard (http, navegador antigo): deixa o codigo selecionado
      // para o participante copiar com o teclado, em vez de falhar em silencio.
      try {
        var alvo = document.querySelector('#vhlab-sessao .codigo');
        var faixa = document.createRange();
        faixa.selectNodeContents(alvo);
        var selecao = window.getSelection();
        selecao.removeAllRanges();
        selecao.addRange(faixa);
        if (document.execCommand('copy')) confirmou();
        else { botao.textContent = 'selecionado'; setTimeout(function () { botao.textContent = original; }, 1600); }
      } catch (_) {}
    }
  }

  // Gatilhos de inicio e fim. Delimitam a captura: sem eles nao ha como distinguir a
  // conversa de quem terminou de proposito da de quem fechou a aba no meio, nem separar um
  // teste do facilitador de uma coleta real feita na mesma aba minutos depois.
  //
  // O servidor identifica a sessao pelo MESMO cookie que ja acompanha as chamadas de /rag,
  // entao nao ha nada a enviar no corpo — o codigo nunca trafega no JavaScript.
  var estado = 'nao_iniciada';

  function avisarServidor(acao) {
    try {
      return fetch('/rag/sessao/' + acao, { method: 'POST' });
    } catch (_) {
      return null;
    }
  }

  function pintarEstado() {
    var rotulo = document.querySelector('#vhlab-sessao .estado');
    var finalizar = document.getElementById('vhlab-finalizar');
    if (!rotulo) return;

    if (estado === 'nao_iniciada') {
      rotulo.textContent = 'nao iniciada';
      rotulo.className = 'estado';
      if (finalizar) finalizar.hidden = true;
    } else if (estado === 'em_andamento') {
      rotulo.textContent = 'em andamento';
      rotulo.className = 'estado';
      if (finalizar) { finalizar.hidden = false; finalizar.disabled = false; }
    } else {
      rotulo.textContent = 'encerrada';
      rotulo.className = 'estado encerrada';
      if (comecar) comecar.hidden = true;
      if (finalizar) { finalizar.hidden = false; finalizar.disabled = true; }
    }
  }

  function comecar() {
    if (estado !== 'nao_iniciada') return;
    estado = 'em_andamento';

    var tela = document.getElementById('vhlab-inicio');
    if (tela) tela.parentNode.removeChild(tela);

    // A barra de progresso do Unity so faz sentido a partir daqui.
    var barra = document.querySelector('#unity-loading-bar');
    if (barra) barra.style.display = 'block';

    // So agora o build comeca a baixar. O clique tambem serve de gesto do usuario, que e o
    // que os navegadores exigem para liberar audio — sem ele a primeira fala do personagem
    // pode sair muda.
    if (typeof window.__vhlabIniciarUnity === 'function') window.__vhlabIniciarUnity();

    pintarEstado();
    avisarServidor('iniciar');
  }

  function finalizar(botaoCopiar) {
    if (estado !== 'em_andamento') return;
    estado = 'encerrada';
    pintarEstado();
    avisarServidor('finalizar');
    // O codigo so serve depois que a conversa acaba; copiar sozinho aqui poupa o
    // participante de procurar o botao no momento em que ele vai para o formulario.
    if (botaoCopiar) copiar(botaoCopiar);
  }

  function montarTelaDeInicio() {
    if (document.getElementById('vhlab-inicio')) return;

    var tela = document.createElement('div');
    tela.id = 'vhlab-inicio';

    // Sem qualquer mencao a genero: o participante nao pode saber em que grupo esta.
    tela.innerHTML =
      '<div class="painel">' +
        '<h1>Conversa com Humano Virtual</h1>' +
        '<ol>' +
          '<li>Você vai conversar com um Humano Virtual sobre audiências públicas da ' +
            'Câmara dos Deputados.</li>' +
          '<li>Copie o código da sessão e cole no questionário.</li>' +
          '<li>Escreva suas perguntas no campo e pressione Enter. O Humano Virtual responde falando.</li>' +
          '<li>Quando terminar, clique em <strong>finalizar</strong> no canto da tela.</li>' +
          
        '</ol>' +
        '<p class="aviso">Ao começar, o ambiente 3D será carregado. ' +
          'Pode levar alguns minutos na primeira vez; mantenha esta aba aberta.</p>' +
      '</div>';

    var botao = document.createElement('button');
    botao.type = 'button';
    botao.textContent = 'Começar';
    botao.addEventListener('click', comecar);
    tela.querySelector('.painel').appendChild(botao);

    (document.body || document.documentElement).appendChild(tela);
  }

  function montar() {
    if (document.getElementById('vhlab-sessao')) return;
    var caixa = document.createElement('div');
    caixa.id = 'vhlab-sessao';

    var texto = document.createElement('div');
    texto.innerHTML = '<div class="rotulo">código da sessão</div>';
    var valor = document.createElement('div');
    valor.className = 'codigo';
    valor.textContent = codigo;
    texto.appendChild(valor);
    caixa.appendChild(texto);

    var rotuloEstado = document.createElement('div');
    rotuloEstado.className = 'estado';
    texto.appendChild(rotuloEstado);

    var copia = document.createElement('button');
    copia.type = 'button';
    copia.textContent = 'copiar código';
    copia.addEventListener('click', function () { copiar(copia); });
    caixa.appendChild(copia);

    var botaoFinalizar = document.createElement('button');
    botaoFinalizar.type = 'button';
    botaoFinalizar.id = 'vhlab-finalizar';
    botaoFinalizar.textContent = 'Finalizar';
    botaoFinalizar.hidden = true;
    botaoFinalizar.addEventListener('click', function () { finalizar(copia); });
    caixa.appendChild(botaoFinalizar);

    if (URL_FORMULARIO) {
      var link = document.createElement('button');
      link.type = 'button';
      link.textContent = 'questionario';
      link.addEventListener('click', function () {
        // Copia antes de abrir: se o formulario nao aceitar preenchimento por URL, o
        // codigo ja esta na area de transferencia para colar no campo.
        copiar(copia);
        window.open(URL_FORMULARIO.split('{ID}').join(encodeURIComponent(codigo)), '_blank', 'noopener');
      });
      caixa.appendChild(link);
    }

    (document.body || document.documentElement).appendChild(caixa);
    pintarEstado();
  }

  function iniciar() { montar(); montarTelaDeInicio(); }

  if (document.body) iniciar();
  else document.addEventListener('DOMContentLoaded', iniciar);
})();
</script>
"""


def main() -> int:
    index = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "site/index.html")
    if not index.is_file():
        print(f"erro: {index} nao encontrado", file=sys.stderr)
        return 1

    html = index.read_text(encoding="utf-8")
    if MARCA in html:
        print("sessao ja injetada")
        return 0

    if "</body>" not in html:
        print("erro: </body> nao encontrado", file=sys.stderr)
        return 1

    if MARCA_BASE not in html:
        if "<head>" not in html:
            print("erro: <head> nao encontrado", file=sys.stderr)
            return 1
        html = html.replace("<head>", "<head>\n" + BASE, 1)

    # Falhar alto: se o template do Unity mudar e a ancora sumir, o build carregaria
    # sozinho e a tela de inicio nunca apareceria — um erro silencioso no meio da coleta.
    if ANCORA_BOOT not in html:
        print(f"erro: ancora do boot do Unity nao encontrada ({ANCORA_BOOT!r})", file=sys.stderr)
        return 1
    html = html.replace(ANCORA_BOOT, GANCHO_BOOT, 1)

    if ANCORA_BARRA in html:
        html = html.replace(ANCORA_BARRA, "", 1)

    # json.dumps produz um literal JS valido tambem para a string vazia (vira "").
    script = MODELO.replace("__URL_FORMULARIO__", json.dumps(os.getenv("URL_FORMULARIO", "")))
    html = html.replace("</body>", script + "  </body>", 1)

    index.write_text(html, encoding="utf-8")
    print(f"sessao injetada em {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
