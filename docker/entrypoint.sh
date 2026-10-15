#!/usr/bin/env sh
# Optionally builds the corpus/index on first start (AUTO_BOOTSTRAP=true), then runs CMD.
set -e
if [ "${AUTO_BOOTSTRAP:-false}" = "true" ]; then
  if [ ! -f "/app/data/index/.bootstrapped" ]; then
    echo "[entrypoint] bootstrapping corpus + index + graph (first start; takes a while on CPU)"
    agvrag bootstrap && mkdir -p /app/data/index && touch /app/data/index/.bootstrapped
  fi
fi
exec "$@"
