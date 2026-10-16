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

# Sets up a fresh Linux machine (Ubuntu; x86_64 or aarch64; with or without an NVIDIA GPU)
# for our training, from a clone of this repo (thud/PLAN.md, the GPU readiness checklist):
#
#   1. system packages and OpenSpiel's dependencies: its install.sh (apt: clang, cmake,
#      python3-dev, ...; clones abseil, pybind11, nlohmann json, libnop, ...), and rsync;
#   2. a venv, venv/, with OpenSpiel's requirements.txt and torch==2.10.0, whose wheel
#      carries LibTorch (on x86_64 PyPI's default wheel is the CUDA build; TORCH_INDEX_URL
#      picks another, e.g. https://download.pytorch.org/whl/cpu);
#   3. the shell lines of CLAUDE.md (the venv, PYTHONPATH) in ~/.bashrc;
#   4. build/ (pyspiel, thud_test) and build-shared/ (libopen_spiel.so with LibTorch);
#   5. our programs into build-shared/ (thud/az/build.sh): the trainer and the tools a
#      GPU run uses;
#   6. hexparrot/thudgame at the commit our references use (thud/experiments/
#      perft_reference.py), for the cross-check and matches against it;
#   7. every check: thud/run_checks.sh --full;
#   8. with a GPU: a CUDA smoke test — LibTorch sees it; the trainer runs 2 steps on
#      cuda:0 and on two copies, cuda:0,cuda:0; az_match plays a pair on cuda:0.
#
# Each step is skipped when its result is there already, so a rerun goes on after a
# failure. Compile jobs follow the memory available (1.5 GB each).
#
#   git clone -b thud https://github.com/lemony-fresh/open_spiel.git ~/thud-openspiel
#   ~/thud-openspiel/thud/setup_machine.sh 2>&1 | tee ~/setup.log
#   [--skip-apt] (no sudo: the packages are there) [--skip-checks] [--gpu-smoke-only]
#   [--no-bashrc] (leave ~/.bashrc alone: a test clone, or a shell set up by hand)
#
# SMOKE_DEVICE=cpu:0 runs step 8's steps on the CPU (without its CUDA check): the smoke
# test's own check, on a machine without a GPU.

set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
skip_apt=0 skip_checks=0 smoke_only=0 no_bashrc=0
for a in "$@"; do
  case $a in
    --skip-apt) skip_apt=1 ;;
    --skip-checks) skip_checks=1 ;;
    --gpu-smoke-only) smoke_only=1 ;;
    --no-bashrc) no_bashrc=1 ;;
    *) echo "Unknown option $a" >&2; exit 2 ;;
  esac
done
step() { echo; echo "=== $(date '+%H:%M:%S') $*"; }
fail() { echo "SETUP FAILED: $*" >&2; exit 1; }
jobs() {
  local mb
  mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
  local j=$((mb / 1536))
  local n
  n=$(nproc)
  [ "$j" -gt "$n" ] && j=$n
  [ "$j" -lt 1 ] && j=1
  echo "$j"
}
gpu=0
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 && gpu=1

gpu_smoke() {
  local dev=${SMOKE_DEVICE:-cuda:0}
  step "8. the smoke test on $dev"
  if [[ $dev == cuda* ]]; then
    python3 -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; \
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))" ||
      fail "LibTorch sees no GPU (the wheel's CUDA against the driver's: nvidia-smi)"
  fi
  local t
  t=$(mktemp -d "${TMPDIR:-/tmp}/gpu_smoke.XXXXXX")
  for devices in "$dev" "$dev,$dev"; do
    local run=$t/run_${devices//[:,]/_}
    build-checks/bin/make_test_network dir="$run" nn_model=resnet_conv_policy input_planes=1 \
      nn_width=32 nn_depth=2 checkpoints=1 > /dev/null 2>&1 || fail "make_test_network"
    OMP_NUM_THREADS=2 build-shared/az_trainer --flagfile=thud/experiments/az_thud.flags \
      --path="$run" --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=32 \
      --nn_depth=2 --devices="$devices" --actors=4 --evaluators=0 --inference_batch_size=4 \
      --inference_threads=1 --max_simulations=16 --replay_buffer_size=2048 \
      --replay_buffer_reuse=4 --train_batch_size=64 --learner_batches=2 --max_steps=2 \
      > "$run.log" 2>&1 || { tail -5 "$run.log"; fail "the trainer on $devices"; }
    grep -q "Running the learner on device 0: $dev" "$run/log-learner.txt" ||
      fail "the trainer on $devices did not learn on $dev"
    [ "$(wc -l < "$run/learner.jsonl")" -eq 2 ] || fail "the trainer on $devices: not 2 steps"
    echo "the trainer: 2 steps on $devices"
  done
  local net=$t/run_${dev//[:,]/_}
  OMP_NUM_THREADS=1 build-shared/az_match a="$net:-1" b="$net:-1" sims=16 pairs=1 \
    devices="$dev" > "$t/match.jsonl" 2> "$t/match.err" ||
    { tail -3 "$t/match.err"; fail "az_match on $dev"; }
  grep -q "\"devices\": \"$dev\"" "$t/match.jsonl" || fail "az_match did not record $dev"
  echo "az_match: a pair on $dev"
  rm -rf "$t"
  echo "The smoke test on $dev passed"
}

if [ "$smoke_only" -eq 1 ]; then
  [ "$gpu" -eq 1 ] || [ -n "${SMOKE_DEVICE:-}" ] || fail "no GPU (nvidia-smi)"
  source venv/bin/activate
  gpu_smoke
  exit 0
fi

step "0. the machine: $(uname -m), $(nproc) cores, $(awk '/^MemTotal:/ {print int($2/1048576)}' /proc/meminfo) GB, $(df -h . | awk 'NR==2 {print $4}') free; GPU: $([ $gpu -eq 1 ] && nvidia-smi -L | head -1 || echo none)"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' ||
  fail "python3 3.10 or newer needed"

step "1. system packages and OpenSpiel's dependencies (install.sh)"
if [ "$skip_apt" -eq 0 ]; then
  sudo apt-get -y update && sudo apt-get -y install git rsync clang cmake python3-venv \
    python3-dev wget || fail "apt-get"
fi
# OpenSpiel's own script; its optional downloads (a LibTorch zip, OR-Tools) are not ours:
# LibTorch comes from the torch wheel (CLAUDE.md, Environment).
if [ ! -f open_spiel/libnop/libnop/include/nop/serializer.h ] || [ ! -d open_spiel/abseil-cpp ]; then
  OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF \
    OPEN_SPIEL_BUILD_WITH_LIBNOP=ON ./install.sh || fail "install.sh"
fi

step "2. the venv, OpenSpiel's requirements and torch 2.10.0"
[ -d venv ] || python3 -m venv venv || fail "python3 -m venv"
source venv/bin/activate
python3 -c 'import torch, absl, numpy' 2> /dev/null || {
  python3 -m pip install --upgrade pip setuptools &&
    python3 -m pip install -r requirements.txt &&
    python3 -m pip install torch==2.10.0 ${TORCH_INDEX_URL:+--index-url "$TORCH_INDEX_URL"}
} || fail "pip"
python3 -c 'import torch; print("torch", torch.__version__, "CUDA build:", torch.version.cuda)'

step "3. the shell environment in ~/.bashrc"
if [ "$no_bashrc" -eq 1 ]; then
  echo "left alone (--no-bashrc)"
elif ! grep -q "$repo/venv/bin/activate" ~/.bashrc 2> /dev/null; then
  cat >> ~/.bashrc <<EOF
# thud-openspiel (thud/setup_machine.sh; CLAUDE.md, Shell environment)
export PYTHONPATH=\$PYTHONPATH:$repo
export PYTHONPATH=\$PYTHONPATH:$repo/build/python
source $repo/venv/bin/activate
EOF
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$repo:$repo/build/python"

step "4. build/ (pyspiel, thud_test) and build-shared/ (libopen_spiel.so with LibTorch)"
torch_prefix=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
mkdir -p build build-shared
if [ ! -f build/Makefile ]; then
  (cd build && CXX=clang++ cmake -DPython3_EXECUTABLE="$(which python3)" \
    -DCMAKE_CXX_COMPILER=clang++ ../open_spiel) || fail "cmake build/"
fi
nice -n 19 make -C build -j"$(jobs)" thud_test pyspiel || fail "make build/"
if [ ! -f build-shared/Makefile ]; then
  (cd build-shared && BUILD_SHARED_LIB=ON BUILD_TYPE=Release OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON \
    OPEN_SPIEL_BUILD_WITH_LIBNOP=ON CXX=clang++ cmake -DPython3_EXECUTABLE="$(which python3)" \
    -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH="$torch_prefix" ../open_spiel) ||
    fail "cmake build-shared/"
fi
nice -n 19 make -C build-shared -j"$(jobs)" open_spiel || fail "make build-shared/"

step "5. our programs into build-shared/"
for main in thud/az/az_trainer.cc thud/az/az_grow.cc thud/az/az_restructure.cc \
    thud/az/az_fresh_optimizer.cc thud/experiments/az_match.cc \
    thud/experiments/az_dwarf_value_probe.cc thud/experiments/az_blunder_check.cc \
    thud/experiments/az_search_cost.cc; do
  name=$(basename "$main" .cc)
  # Rebuilt when it or any source of our AlphaZero copy is newer than the program.
  newest=$(ls -t "$main" thud/az/*.cc thud/az/*.h thud/experiments/*.h | head -1)
  if [ ! -x "build-shared/$name" ] || [ "$newest" -nt "build-shared/$name" ]; then
    thud/az/build.sh "$main" || fail "building $name"
  fi
done

step "6. hexparrot/thudgame (thud/experiments/perft_reference.py)"
hexparrot=$(python3 -c "import sys; sys.path.insert(0, 'thud/experiments'); \
import perft_reference as r; print(r.DEFAULT_HEXPARROT, r.HEXPARROT_COMMIT)")
read -r hex_dir hex_commit <<< "$hexparrot"
if [ ! -d "$hex_dir/.git" ]; then
  git clone -q https://github.com/hexparrot/thudgame.git "$hex_dir" || fail "cloning hexparrot"
fi
git -C "$hex_dir" checkout -q "$hex_commit" || fail "hexparrot's commit $hex_commit"

if [ "$skip_checks" -eq 0 ]; then
  step "7. every check (thud/run_checks.sh --full)"
  thud/run_checks.sh --full || fail "a check (build-checks/last/)"
fi

if [ "$gpu" -eq 1 ]; then
  gpu_smoke
else
  step "8. no GPU here: the CUDA smoke test skipped"
fi
step "The machine is set up."
