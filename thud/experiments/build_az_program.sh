#!/bin/bash
# Copyright 2026 The Thud-on-OpenSpiel authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Builds one of our C++ programs that use OpenSpiel's C++ AlphaZero,
# thud/experiments/PROGRAM.cc, into build-shared/PROGRAM, without touching OpenSpiel's
# CMake files: it links against OpenSpiel built as a shared library, as OpenSpiel's
# docs/library.md describes, and compiles upstream's AlphaZero model and evaluator
# (model.cc, vpnet.cc, vpevaluator.cc, unmodified) alongside, since that library leaves
# them out. The compiler flags are those CMake uses for them in build-torch/.
#
#   thud/experiments/build_az_program.sh az_layout_check    # or az_throughput
#
# Needs build-shared/libopen_spiel.so, from the repo root:
#
#   mkdir -p build-shared && cd build-shared
#   BUILD_SHARED_LIB=ON BUILD_TYPE=Release OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON \
#     OPEN_SPIEL_BUILD_WITH_LIBNOP=ON CXX=clang++ cmake \
#     -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=clang++ \
#     -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') \
#     ../open_spiel
#   make -j10 open_spiel    # ~2.5 min

set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: $0 PROGRAM   (builds thud/experiments/PROGRAM.cc)" >&2
  exit 1
fi
program=$1
repo=$(cd "$(dirname "$0")/../.." && pwd)
src=$repo/open_spiel
lib=$repo/build-shared
torch=$(python3 -c 'import os, torch; print(os.path.dirname(torch.__file__))')

if [ ! -f "$lib/libopen_spiel.so" ]; then
  echo "Missing $lib/libopen_spiel.so: build it first (see this script's comments)." >&2
  exit 1
fi

clang++ -std=gnu++20 -O3 -DNDEBUG -Wno-everything \
  -DOPEN_SPIEL_BUILD_WITH_LIBNOP -DOPEN_SPIEL_BUILD_WITH_LIBTORCH \
  -DOPEN_SPIEL_BUILD_WITH_PYTHON -DOPEN_SPIEL_ENABLE_JAX -DOPEN_SPIEL_ENABLE_PYTORCH \
  -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE \
  -I"$repo" -I"$src" -I"$src/abseil-cpp" -I"$src/json/include" \
  -I"$src/libnop/libnop/include" \
  -isystem "$torch/include" -isystem "$torch/include/torch/csrc/api/include" \
  -o "$lib/$program" \
  "$repo/thud/experiments/$program.cc" \
  "$src/algorithms/alpha_zero_torch/model.cc" \
  "$src/algorithms/alpha_zero_torch/vpnet.cc" \
  "$src/algorithms/alpha_zero_torch/vpevaluator.cc" \
  -L"$lib" -lopen_spiel -Wl,-rpath,"$lib" \
  -Wl,--no-as-needed "$torch/lib/libtorch.so" "$torch/lib/libtorch_cpu.so" \
  "$torch/lib/libc10.so" -Wl,--as-needed -Wl,-rpath,"$torch/lib"
echo "Built $lib/$program"
