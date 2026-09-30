#!/usr/bin/env python3
"""Injeta um overlay de diagnostico no index.html gerado pelo Unity.

O index.html e regerado a cada build, entao esta injecao roda no sync-build.sh, depois
da copia. Objetivo: no celular nao ha console acessivel, e o template padrao do Unity
falha em silencio (canvas some -> body branco) ou usa alert(), que some ao ser fechado.
Aqui qualquer erro, perda de contexto WebGL ou banner do proprio Unity vira texto legivel
na tela do aparelho.
"""
import pathlib
import sys

MARCA = "<!-- vhlab-diagnostics -->"

SCRIPT = """<!-- vhlab-diagnostics -->
<script>
(function () {
  var box = null;
  function show(kind, msg) {
    try {
      if (!box) {
        box = document.createElement('div');
        box.style.cssText = 'position:fixed;inset:0;z-index:99999;background:#111;color:#eee;'
          + 'font:13px/1.5 -apple-system,system-ui,sans-serif;padding:16px;overflow:auto;'
          + 'white-space:pre-wrap;-webkit-user-select:text;user-select:text';
        (document.body || document.documentElement).appendChild(box);
      }
      box.textContent += '[' + kind + '] ' + msg + '\\n\\n';
    } catch (_) { /* nada a fazer se nem isso funcionar */ }
  }
  window.__vhlabShow = show;

  window.addEventListener('error', function (e) {
    show('erro', (e.message || 'erro sem mensagem')
      + (e.filename ? '\\n' + e.filename + ':' + e.lineno : ''));
  });
  window.addEventListener('unhandledrejection', function (e) {
    var r = e.reason;
    show('promise', String((r && r.message) || r));
  });

  function hookCanvas() {
    var c = document.querySelector('#unity-canvas');
    if (!c) return false;
    c.addEventListener('webglcontextlost', function (e) {
      show('webgl', 'contexto WebGL PERDIDO - tipicamente falta de memoria no aparelho');
    }, false);
    c.addEventListener('webglcontextrestored', function () {
      show('webgl', 'contexto WebGL restaurado');
    }, false);
    return true;
  }
  if (!hookCanvas()) document.addEventListener('DOMContentLoaded', hookCanvas);

  // O Unity chama unityShowBanner para erros proprios (ex.: "Out of memory").
  var original = window.unityShowBanner;
  window.unityShowBanner = function (msg, type) {
    show('unity/' + (type || 'info'), msg);
    if (typeof original === 'function') { try { original(msg, type); } catch (_) {} }
  };

  // Relatorio periodico: se a tela ficar branca, o ultimo estado registrado indica onde parou.
  var ticks = 0;
  var timer = setInterval(function () {
    ticks++;
    var c = document.querySelector('#unity-canvas');
    var vivo = false;
    try {
      var gl = c && (c.getContext('webgl2') || c.getContext('webgl'));
      vivo = !!gl && !gl.isContextLost();
    } catch (_) {}
    var m = window.performance && performance.memory;
    var estado = 'canvas=' + (c ? c.width + 'x' + c.height + ' visivel=' + !!c.offsetParent : 'AUSENTE')
      + ' gl=' + (vivo ? 'ok' : 'PERDIDO')
      + (m ? ' heap=' + (m.usedJSHeapSize / 1048576).toFixed(0) + 'MB' : '');
    // So mostra quando algo deu errado; caso contrario apenas guarda.
    window.__vhlabUltimoEstado = estado;
    if (!c || !vivo) { show('estado', estado); clearInterval(timer); }
    if (ticks > 120) clearInterval(timer);
  }, 1000);

  // Toque com 3 dedos mostra o estado atual, para inspecao manual no aparelho.
  window.addEventListener('touchstart', function (e) {
    if (e.touches && e.touches.length === 3) show('estado', window.__vhlabUltimoEstado || 'sem leitura');
  });
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
        print("diagnostico ja injetado")
        return 0

    # O template mostra falha fatal com alert(), que o usuario fecha e perde. Redireciona
    # para o overlay, que permanece na tela e pode ser lido/copiado.
    html = html.replace("alert(message);", "window.__vhlabShow('unity/fatal', message);")

    if "</body>" not in html:
        print("erro: </body> nao encontrado", file=sys.stderr)
        return 1
    html = html.replace("</body>", SCRIPT + "  </body>", 1)

    index.write_text(html, encoding="utf-8")
    print(f"diagnostico injetado em {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
