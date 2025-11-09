#!/usr/bin/env bash
set -euo pipefail

# 사용법: ./runtb.sh Pkg::mkTop [sim-args...]
# 예:     ./runtb.sh TbTimeStampGenerator::mkTbTimeStampGenerator +bscvcd
TOP="${1:?usage: $0 <Pkg::mkTop> [sim-args...]}"
shift || true

# --- bsc / 라이브러리 경로 ---
BSC_ROOT="${BSC_ROOT:-$HOME/bsc/inst}"
export PATH="$BSC_ROOT/bin:$PATH"

# BLUESPECDIR 미지정이면 자동 추론 (Libraries 우선)
if [ -z "${BLUESPECDIR:-}" ]; then
  for d in "$BSC_ROOT/lib/Libraries" "$BSC_ROOT/lib"; do
    [ -d "$d" ] && { export BLUESPECDIR="$d"; break; }
  done
fi

command -v bsc >/dev/null 2>&1 || { echo "ERROR: bsc not found in PATH"; exit 127; }
[ -d "${BLUESPECDIR:-/nope}" ] || { echo "ERROR: BLUESPECDIR not a dir: ${BLUESPECDIR:-<unset>}"; exit 128; }

# --- 경로 고정 ---
cd "$(dirname "$0")"
mkdir -p build

PKG="${TOP%%::*}"          # 예) TbTimeStampGenerator
TOPMOD="${TOP#*::}"        # 예) mkTbTimeStampGenerator
TBFILE="tb/${PKG}.bsv"     # 예) tb/TbTimeStampGenerator.bsv
[ -f "$TBFILE" ] || { echo "ERROR: TB file not found: $TBFILE"; exit 2; }

# --- 컴파일 & 엘라보 ---
bsc -u -elab -sim \
    -bdir build -simdir build -info-dir build \
    -p +:$BLUESPECDIR:src:tb \
    -g "$TOPMOD" "$TBFILE"

# --- 링크 ---
bsc -sim \
    -bdir build -simdir build -info-dir build \
    -p +:$BLUESPECDIR:src:tb \
    -e "$TOPMOD" -o build/sim.out

# --- 실행 ---
exec ./build/sim.out "$@"
