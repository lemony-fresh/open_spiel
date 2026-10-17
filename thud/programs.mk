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

# Builds the programs our runs use into build-shared/, incrementally (the user,
# 2026-10-17: a stale az_capture_targets, built before the input planes, aborted a night's
# probe): each source compiled once into build-shared/obj-programs/, again only when it or
# a header it includes changed (-MMD), so a build with nothing to do takes a second.
# thud/build_programs.sh calls it (and refuses while one of these programs runs); by hand:
#
#   make -f thud/programs.mk -j10 [PROGRAM ...]
#
# The flags are thud/az/build.sh's (-O3 -DNDEBUG: run speed; the tests build without
# -DNDEBUG in build-tests/, thud/tests.mk). build.sh stays for one-off programs. Not here:
# az_trainer_d1 (the frozen binary of runs N, N2 and their continuation), and the
# programs on upstream's AlphaZero (thud/experiments/build_az_program.sh).

REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
LIB := $(REPO)/build-shared
OBJ := $(LIB)/obj-programs
SRC := $(REPO)/open_spiel
TORCH := $(shell python3 -c 'import importlib.util; print(importlib.util.find_spec("torch").submodule_search_locations[0])')
CXX := clang++

COMMON := -std=gnu++20 -O3 -DNDEBUG -MMD -MP -I$(REPO) -I$(SRC) -I$(SRC)/abseil-cpp \
  -I$(SRC)/json/include -I$(SRC)/libnop/libnop/include -DOPEN_SPIEL_BUILD_WITH_LIBNOP
TORCH_FLAGS := $(COMMON) -Wno-everything -DOPEN_SPIEL_BUILD_WITH_LIBTORCH \
  -DOPEN_SPIEL_BUILD_WITH_PYTHON -DOPEN_SPIEL_ENABLE_JAX -DOPEN_SPIEL_ENABLE_PYTORCH \
  -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE \
  -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LINK := -L$(LIB) -lopen_spiel -Wl,-rpath,$(LIB) \
  -Wl,--start-group $(wildcard $(LIB)/abseil-cpp/absl/*/libabsl_*.a) -Wl,--end-group
TORCH_CUDA := $(wildcard $(TORCH)/lib/libtorch_cuda.so $(TORCH)/lib/libc10_cuda.so)
TORCH_LINK := $(LINK) -Wl,--no-as-needed $(TORCH)/lib/libtorch.so \
  $(TORCH)/lib/libtorch_cpu.so $(TORCH)/lib/libc10.so $(TORCH_CUDA) -Wl,--as-needed \
  -Wl,-rpath,$(TORCH)/lib

COPIES := $(addprefix $(OBJ)/az/,mcts.o alpha_zero.o model.o vpnet.o vpevaluator.o)
AZ_PROGRAMS := az_trainer az_grow az_restructure az_fresh_optimizer
EXPERIMENT_PROGRAMS := az_match az_forgetting az_merge_buffers az_target_quality \
  az_head_probe az_reference_pilot az_noise_probe az_search_cost az_blunder_probe \
  az_capture_targets az_teacher_stats az_feature_probe az_long_capture_probe \
  az_input_plane_trace az_risk_probe az_dwarf_value_probe az_unsafe_targets
PLAIN_PROGRAMS := az_buffer_truncate

ALL := $(addprefix $(LIB)/,$(AZ_PROGRAMS) $(EXPERIMENT_PROGRAMS) $(PLAIN_PROGRAMS))

all: $(ALL)
$(AZ_PROGRAMS) $(EXPERIMENT_PROGRAMS) $(PLAIN_PROGRAMS): %: $(LIB)/%
.PHONY: all $(AZ_PROGRAMS) $(EXPERIMENT_PROGRAMS) $(PLAIN_PROGRAMS)

$(OBJ)/az/%.o: $(REPO)/thud/az/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(TORCH_FLAGS) -c $< -o $@

$(OBJ)/experiments/%.o: $(REPO)/thud/experiments/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(TORCH_FLAGS) -c $< -o $@

$(OBJ)/plain/%.o: $(REPO)/thud/experiments/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(COMMON) -c $< -o $@

$(addprefix $(LIB)/,$(AZ_PROGRAMS)): $(LIB)/%: $(OBJ)/az/%.o $(COPIES)
	nice -n 19 $(CXX) -o $@ $^ $(TORCH_LINK)

$(addprefix $(LIB)/,$(EXPERIMENT_PROGRAMS)): $(LIB)/%: $(OBJ)/experiments/%.o $(COPIES)
	nice -n 19 $(CXX) -o $@ $^ $(TORCH_LINK)

$(addprefix $(LIB)/,$(PLAIN_PROGRAMS)): $(LIB)/%: $(OBJ)/plain/%.o
	nice -n 19 $(CXX) -o $@ $^ $(LINK)

# A program must be relinked when libopen_spiel.so (the game) changes.
$(ALL): $(LIB)/libopen_spiel.so

-include $(shell find $(OBJ) -name '*.d' 2>/dev/null)
