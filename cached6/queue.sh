#!/usr/bin/env bash
set -eu
phase=${1:-train}
case "$phase" in train|eval) ;; *) echo 'expected train or eval' >&2; exit 2;; esac
: "${RUN_ROOT:?set an independent output root for the six cached methods}"
: "${CUDA_VISIBLE_DEVICES:?set the assigned GPU0 UUID}"
PY=${PY:-/data/specenv/bin/python}
runner="$(cd "$(dirname "$0")" && pwd)/runner.py"
mkdir -p "$RUN_ROOT"
one() {
    "$PY" -u "$runner" "$phase" --arm "$1" --out "$RUN_ROOT/arm-$1" >> "$RUN_ROOT/$phase-$1.log" 2>&1
}
if [ "$phase" = train ]; then
    one 7 & p7=$!
    one 8 & p8=$!
    status=0
    wait "$p7" || status=$?
    wait "$p8" || status=$?
    [ "$status" -eq 0 ] || exit "$status"
else
    one 7
    one 8
fi
for arm in 9 14 15 22; do one "$arm"; done
