#!/usr/bin/env bash
# tail-chat.sh — acompanha as conversas na Polaris, só o que interessa.
#
# Mostra, por turno: quantas mensagens, o tamanho de cada papel, e o comando que
# o modelo pediu. Sem isso a análise vira "me manda o log" a cada teste.
#
#   ./tail-chat.sh          # segue ao vivo
#   ./tail-chat.sh 20       # as últimas 20 entradas e sai
HOST="${POLARIS_HOST:-blacksun}"
LOG='~/dev/polaris/polaris-v3-api/polaris.log'
N="${1:-}"
FILTER='grep --line-buffered -E "📜 chat \(|\[assistant\] .*tool_call|🛠️ Tool call|decode: [0-9]+ toks"'
if [[ -n "$N" ]]; then
  ssh "$HOST" "tail -n 4000 $LOG | $FILTER | tail -n $N"
else
  ssh "$HOST" "tail -f -n 40 $LOG | $FILTER"
fi
