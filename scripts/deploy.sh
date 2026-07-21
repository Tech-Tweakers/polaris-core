#!/usr/bin/env bash
# deploy.sh — leva o que está no GitHub para a blacksun e reinicia o que precisa.
#
# A regra: o código NASCE no repo local, sobe por push, e a blacksun só CONSOME.
# Ninguém edita arquivo remoto na mão. Foi assim que a gente ficou horas com um
# binário descasado (20/07) e quase editou um polaris_bind.cpp de fevereiro
# achando que era o atual (21/07) — três cópias, nenhuma autoritativa.
#
#   ./deploy.sh              # api  (pull + restart)
#   ./deploy.sh core         # bind (pull + compila + restart)
#   ./deploy.sh all          # os dois
#   ./deploy.sh api --no-restart
set -euo pipefail
# shellcheck disable=SC2029
# SERVICE and HOST are intentionally expanded on the client side before SSH.

HOST="${POLARIS_HOST:-blacksun}"
SERVICE="${POLARIS_SERVICE:-polaris-api}"
TARGET="${1:-api}"
RESTART=1
[[ "${2:-}" == "--no-restart" ]] && RESTART=0

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
ok()  { printf '  \033[32m✓\033[0m %s\n' "$*"; }

# Aviso, não bloqueio: às vezes se quer subir só um dos repos.
check_local_pushed() {
  local dir="$1" name="$2"
  [[ -d "$dir" ]] || return 0
  if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
    printf '  \033[33m!\033[0m %s tem mudança não commitada — ela NÃO vai subir\n' "$name"
  fi
  local br; br=$(git -C "$dir" rev-parse --abbrev-ref HEAD)
  if [[ -n "$(git -C "$dir" log "origin/$br..HEAD" --oneline 2>/dev/null)" ]]; then
    printf '  \033[33m!\033[0m %s tem commit não pushado em %s\n' "$name" "$br"
  fi
}

deploy_api() {
  say "polaris-v3-api → $HOST"
  check_local_pushed ~/dev/polaris/polaris-v3-api polaris-v3-api
  ssh "$HOST" 'cd ~/dev/polaris/polaris-v3-api && git pull --ff-only' | sed 's/^/  /'
  ok "código atualizado"
  if (( RESTART )); then
    # shellcheck disable=SC2029
    ssh "$HOST" "sudo systemctl restart ${SERVICE}" && ok "$SERVICE reiniciado"
    sleep 3
    if curl -sf "http://$HOST:8000/v1/models" >/dev/null 2>&1; then
      ok "respondendo em /v1/models"
    else
      printf '  \033[31m✗\033[0m não respondeu — veja: ssh %s "journalctl -u %s -n 40"\n' "$HOST" "$SERVICE"
      return 1
    fi
  fi
}

deploy_core() {
  say "polaris-core (bind C++) → $HOST"
  check_local_pushed ~/dev/polaris/polaris-core polaris-core
  ssh "$HOST" 'cd ~/dev/polaris/polaris-core && git pull --ff-only' | sed 's/^/  /'
  # O .so é carregado pelo processo da API: sem recompilar E reiniciar, o pull
  # não muda nada — o binário velho segue em memória.
  ssh "$HOST" 'cd ~/dev/polaris/polaris-core && cmake --build build -j4 2>&1 | tail -3' | sed 's/^/  /'
  ok "bind compilado"
  # O .so compilado em polaris-core/build/ NÃO é o que a API carrega: ela lê de
  # polaris_api/polaris_core/{gpu,cpu}/. Sem esta cópia o build sobe e nada muda
  # — foi assim que o generate_chat ficou uma manhã inteira sem efeito (21/07),
  # com o fallback achatando a conversa e mascarando o problema.
  ssh "$HOST" 'set -e
    SRC=~/dev/polaris/polaris-core/build/polaris_core.cpython-310-x86_64-linux-gnu.so
    for D in gpu cpu; do
      DST=~/dev/polaris/polaris-v3-api/polaris_api/polaris_core/$D/polaris_core.cpython-310-x86_64-linux-gnu.so
      [ -f "$DST" ] && cp "$SRC" "$DST" && echo "  .so → $D/"
    done' | sed 's/^/  /'
  ok "bind instalado onde a API carrega"
  if (( RESTART )); then
    # shellcheck disable=SC2029
    ssh "$HOST" "sudo systemctl restart $SERVICE" && ok "$SERVICE reiniciado (carrega o .so novo)"
  fi
}

case "$TARGET" in
  api)  deploy_api ;;
  core) deploy_core ;;
  all)  deploy_core; RESTART=0; deploy_api; # shellcheck disable=SC2029
        ssh "$HOST" "sudo systemctl restart $SERVICE" && ok "reiniciado" ;;
  *)    echo "uso: $0 [api|core|all] [--no-restart]"; exit 1 ;;
esac
say "pronto."
