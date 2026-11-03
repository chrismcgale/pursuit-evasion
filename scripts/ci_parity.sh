#!/usr/bin/env bash
# Every cross-runtime parity check CI runs, in one place, so adding a check
# does not need a workflow edit. Run from the repo root after building cpp/.
#   bash scripts/ci_parity.sh            (CI: replaces the inline loop)
# Each block prints both runtimes' lines and fails on the first mismatch.
set -euo pipefail
tmp=$(mktemp -d)
# `uv run` by default; PE_VENV=.venv runs the venv's binaries directly instead
run() { if [[ -n ${PE_VENV:-} ]]; then "$PE_VENV/bin/$1" "${@:2}"; else uv run "$@"; fi; }

check() {   # check <label> <python args...> -- <cpp args...>
  local label=$1; shift
  local py=() cpp=()
  while [[ $1 != -- ]]; do py+=("$1"); shift; done; shift
  cpp=("$@")
  echo "== $label"
  run pe-parity "${py[@]}" | tee "$tmp/py.txt"
  (cd cpp && ./build/pe_run --parity "${cpp[@]}") | tee "$tmp/cpp.txt"
  run python scripts/check_parity.py "$tmp/py.txt" "$tmp/cpp.txt"
}

# Every game, both stacks: scripted-vs-scripted covers dynamics, capture,
# breach and both scripted laws; bt_safe covers features, the gate (py_trees vs
# the shipped BT.CPP XML), the shield and the committed ONNX policy.
for game in tag assault escort; do
  for ctrl in scripted bt_safe; do
    check "$game $ctrl" --game "$game" --controller "$ctrl" -- --game "$game" --controller "$ctrl"
  done
done

# Own-team separation (off by default) through the same stack. Tag fires it on
# every start; defenders rarely meet, so escort uses a wide 5 m to exercise it.
for spec in "tag 1.5" "escort 5"; do
  set -- $spec
  check "$1 bt_safe separation=$2" --game "$1" --controller bt_safe --separation "$2" \
    -- --game "$1" --controller bt_safe --separation "$2"
done
# Degraded link on the production stack: Python writes the corruption tape it
# draws (env/link.py:write_tape), C++ replays it; every game x stack must
# still match exactly.
for game in tag assault escort; do
  for ctrl in scripted bt_safe; do
    check "$game $ctrl link=vicon_lab" --game "$game" --controller "$ctrl" \
      --link vicon_lab --tape "$tmp/tape.txt" \
      -- --game "$game" --controller "$ctrl" --link vicon_lab --link-tape "$tmp/tape.txt"
  done
done
echo "all parity checks exact"
