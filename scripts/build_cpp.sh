#!/usr/bin/env bash
# Phase 10 C++ 컴포넌트 빌드 + 단위 테스트. 저장소 루트 어디서든 실행 가능.
#   scripts/build_cpp.sh            # Release 빌드 → cpp/build/, ctest 실행
# 산출물: cpp/build/libpt3d_core.a, cpp/build/pt3d_tests, cpp/build/pt3d_cpp.cpython-*.so
# 파이썬에서는 perceptrack3d.geometry.projection_cpp 가 cpp/build 를 sys.path 에 넣어 모듈을 찾는다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
cmake --build "$ROOT/cpp/build" -j"$(nproc)"
ctest --test-dir "$ROOT/cpp/build" --output-on-failure
