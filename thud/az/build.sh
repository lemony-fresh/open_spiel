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

# Builds a program on our copy of OpenSpiel's C++ AlphaZero (thud/az/, namespace
# open_spiel::thud_az) into build-shared/: MAIN.cc plus every copied source, linked
# against OpenSpiel built as a shared library (docs/library.md; see
# thud/experiments/build_az_program.sh for how to build it) and LibTorch, with the
# compiler flags CMake uses for the AlphaZero sources in build-torch/. Runs at the lowest
# CPU priority, so that running experiments hardly notice it. Abseil's static libraries
# follow libopen_spiel.so, which does not re-export the flag parsing the trainer needs
# (CMake adds them for its own targets); the linker takes from them only what is still
# missing.
#
#   thud/az/build.sh thud/az/az_trainer.cc          # -> build-shared/az_trainer
#   thud/az/build.sh thud/az/identity_check.cc      # -> build-shared/identity_check
#
# A MAIN outside thud/az/ may also compile upstream sources: list them after it.

set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 MAIN.cc [MORE_SOURCES.cc ...]" >&2
  exit 1
fi
main=$1
shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
src=$repo/open_spiel
lib=$repo/build-shared
torch=$(python3 -c 'import os, torch; print(os.path.dirname(torch.__file__))')
name=$(basename "$main" .cc)

if [ ! -f "$lib/libopen_spiel.so" ]; then
  echo "Missing $lib/libopen_spiel.so: see thud/experiments/build_az_program.sh." >&2
  exit 1
fi

copies=()
for f in "$repo"/thud/az/{mcts,alpha_zero,model,vpnet,vpevaluator}.cc; do copies+=("$f"); done

nice -n 19 clang++ -std=gnu++20 -O3 -DNDEBUG -Wno-everything \
  -DOPEN_SPIEL_BUILD_WITH_LIBNOP -DOPEN_SPIEL_BUILD_WITH_LIBTORCH \
  -DOPEN_SPIEL_BUILD_WITH_PYTHON -DOPEN_SPIEL_ENABLE_JAX -DOPEN_SPIEL_ENABLE_PYTORCH \
  -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE \
  -I"$repo" -I"$src" -I"$src/abseil-cpp" -I"$src/json/include" \
  -I"$src/libnop/libnop/include" \
  -isystem "$torch/include" -isystem "$torch/include/torch/csrc/api/include" \
  -o "$lib/$name" "$main" "${copies[@]}" "$@" \
  -L"$lib" -lopen_spiel -Wl,-rpath,"$lib" \
  -Wl,--start-group "$lib"/abseil-cpp/absl/*/libabsl_*.a -Wl,--end-group \
  -Wl,--no-as-needed "$torch/lib/libtorch.so" "$torch/lib/libtorch_cpu.so" \
  "$torch/lib/libc10.so" -Wl,--as-needed -Wl,-rpath,"$torch/lib"
echo "Built $lib/$name"
