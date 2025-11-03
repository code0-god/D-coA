#!/usr/bin/env bash
set -euo pipefail

# 사용법: ./runtb.sh <TopModule> [sim-args...]
TOP="${1:?usage: $0 <TopModule> [sim-args...]}"  # 예: TbGenerateTimestamp::mkTbTimestampGenerator

# ---- 1) bsc 실행파일 탐색 (bin, inst/bin, 시스템 PATH, 흔한 경로)
BSC_ROOT="${BSC_ROOT:-$HOME/bsc}"
BSC=""

_try() { [ -n "${1:-}" ] && [ -x "$1" ] && BSC="$1"; }

_try "$BSC_ROOT/bin/bsc"            || true
[ -z "$BSC" ] && _try "$BSC_ROOT/inst/bin/bsc"     || true
[ -z "$BSC" ] && _try "$(command -v bsc 2>/dev/null || true)" || true
[ -z "$BSC" ] && _try "$(find "$BSC_ROOT" ~ /opt /usr/local -maxdepth 6 -type f -name bsc -perm -u+x 2>/dev/null | head -n1)" || true

if [ -z "$BSC" ]; then
  echo "ERROR: bsc executable not found. Set BSC_ROOT to your Bluespec install root (contains bin/ or inst/bin)." >&2
  exit 127
fi

# ---- 2) BLUESPECDIR 결정 (…/lib/Prelude 가 있어야 함)
if [ -z "${BLUESPECDIR:-}" ]; then
  for d in \
    "$(dirname "$BSC")/../lib" \
    "$BSC_ROOT/lib" \
    "$BSC_ROOT/inst/lib"
  do
    if [ -d "$d/Prelude" ]; then
      export BLUESPECDIR="$d"
      break
    fi
  done
fi
if [ ! -d "${BLUESPECDIR:-}/Prelude" ]; then
  echo "ERROR: BLUESPECDIR not set or Prelude missing. Set BLUESPECDIR to a directory that contains Prelude/." >&2
  exit 128
fi

# ---- 3) 어디서 실행해도 hw 기준으로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p build

# ---- 4) 컴파일 + 엘라보 + 실행
"$BSC" -u -elab -sim -p +:src:tb -g "$TOP" src/*.bsv tb/*.bsv
"$BSC" -sim -p +:src:tb -e "$TOP" -o build/sim.out
exec ./build/sim.out "${@:2}"
