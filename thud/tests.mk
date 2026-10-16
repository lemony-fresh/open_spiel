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

# Builds our tests, and the programs the end-to-end tests run, into build-tests/
# (git ignores build*/), incrementally: each source is compiled once into an object, and
# again only when it or a header it includes changed (-MMD). thud/run_tests.sh calls it;
# by hand, from the repo root:
#
#   make -f thud/tests.mk -j10 [TARGET ...]
#
# The flags are thud/az/build.sh's (CMake's for the AlphaZero sources in build-torch/),
# but without -DNDEBUG, so that assert and SPIEL_DCHECK run (CLAUDE.md: tests
# build without it), and at -O2. -Wno-everything is build.sh's too, still to be narrowed
# (CLAUDE.md, Conventions); the tests without LibTorch build with the default warnings,
# as their headers say. The programs here are for the tests only: runs use build-shared/.

REPO := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
OUT := $(REPO)/build-tests
SRC := $(REPO)/open_spiel
LIB := $(REPO)/build-shared
# Found without importing torch (an import takes ~0.7 s).
TORCH := $(shell python3 -c 'import importlib.util; print(importlib.util.find_spec("torch").submodule_search_locations[0])')
CXX := clang++

COMMON := -std=gnu++20 -O2 -MMD -MP -I$(REPO) -I$(SRC) -I$(SRC)/abseil-cpp \
  -I$(SRC)/json/include -I$(SRC)/libnop/libnop/include -DOPEN_SPIEL_BUILD_WITH_LIBNOP
TORCH_FLAGS := $(COMMON) -Wno-everything -DOPEN_SPIEL_BUILD_WITH_LIBTORCH \
  -DOPEN_SPIEL_BUILD_WITH_PYTHON -DOPEN_SPIEL_ENABLE_JAX -DOPEN_SPIEL_ENABLE_PYTORCH \
  -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE \
  -isystem $(TORCH)/include -isystem $(TORCH)/include/torch/csrc/api/include
LINK := -L$(LIB) -lopen_spiel -Wl,-rpath,$(LIB) \
  -Wl,--start-group $(wildcard $(LIB)/abseil-cpp/absl/*/libabsl_*.a) -Wl,--end-group
# LibTorch's CUDA libraries too when the wheel has them (as thud/az/build.sh).
TORCH_CUDA := $(wildcard $(TORCH)/lib/libtorch_cuda.so $(TORCH)/lib/libc10_cuda.so)
TORCH_LINK := $(LINK) -Wl,--no-as-needed $(TORCH)/lib/libtorch.so \
  $(TORCH)/lib/libtorch_cpu.so $(TORCH)/lib/libc10.so $(TORCH_CUDA) -Wl,--as-needed \
  -Wl,-rpath,$(TORCH)/lib

# Our copy of the C++ AlphaZero, linked into every LibTorch program (as build.sh does).
COPIES := $(addprefix $(OUT)/obj/az/,mcts.o alpha_zero.o model.o vpnet.o vpevaluator.o)
# Upstream's, for identity_test only.
UPSTREAM := $(addprefix $(OUT)/obj/upstream/,model.o vpnet.o vpevaluator.o)

TORCH_TESTS := alphabeta_training_test augmentation_test conv_policy_test grow_test \
  learning_settings_test legal_policy_test playout_cap_test \
  growing_buffer_test input_planes_test instrumentation_test kickstart_test \
  pruning_test search_settings_test untried_move_test make_test_network
TORCH_PROGRAMS := az_trainer az_grow az_restructure az_fresh_optimizer
EXPERIMENT_PROGRAMS := az_match teacher_mix_test az_search_cost az_dwarf_value_probe
PLAIN_TESTS := features_test replay_buffer_test
PLAIN_EXPERIMENTS := az_buffer_truncate position_edits_test

# The tests without LibTorch again under AddressSanitizer and UndefinedBehaviorSanitizer
# (CLAUDE.md, Conventions: memory handling; -DNDEBUG_SANITIZER keeps abseil's hash tables
# laid out as in the uninstrumented libopen_spiel.so).
SANITIZE := -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all \
  -DNDEBUG_SANITIZER -g
SANITIZED := $(addprefix $(OUT)/bin/sanitized/,$(PLAIN_TESTS) $(PLAIN_EXPERIMENTS))

ALL := $(addprefix $(OUT)/bin/,$(TORCH_TESTS) $(TORCH_PROGRAMS) $(EXPERIMENT_PROGRAMS) \
  identity_test $(PLAIN_TESTS) $(PLAIN_EXPERIMENTS)) $(SANITIZED)

all: $(ALL)

# Every target can also be named without its folder: make -f thud/tests.mk grow_test
UNSANITIZED := $(filter-out $(SANITIZED),$(ALL))
$(notdir $(UNSANITIZED)): %: $(OUT)/bin/%
sanitized: $(SANITIZED)
.PHONY: all sanitized $(notdir $(UNSANITIZED))

$(OUT)/obj/az/%.o: $(REPO)/thud/az/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(TORCH_FLAGS) -c $< -o $@

$(OUT)/obj/experiments/%.o: $(REPO)/thud/experiments/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(TORCH_FLAGS) -c $< -o $@

$(OUT)/obj/upstream/%.o: $(SRC)/algorithms/alpha_zero_torch/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(TORCH_FLAGS) -c $< -o $@

$(OUT)/obj/plain/%.o: $(REPO)/thud/az/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(COMMON) -c $< -o $@

$(OUT)/obj/plain/%.o: $(REPO)/thud/experiments/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(COMMON) -c $< -o $@

$(OUT)/obj/sanitized/%.o: $(REPO)/thud/az/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(COMMON) $(SANITIZE) -c $< -o $@

$(OUT)/obj/sanitized/%.o: $(REPO)/thud/experiments/%.cc
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(COMMON) $(SANITIZE) -c $< -o $@

$(addprefix $(OUT)/bin/,$(TORCH_TESTS) $(TORCH_PROGRAMS)): $(OUT)/bin/%: $(OUT)/obj/az/%.o $(COPIES)
	@mkdir -p $(@D)
	nice -n 19 $(CXX) -o $@ $^ $(TORCH_LINK)

$(addprefix $(OUT)/bin/,$(EXPERIMENT_PROGRAMS)): $(OUT)/bin/%: $(OUT)/obj/experiments/%.o $(COPIES)
	@mkdir -p $(@D)
	nice -n 19 $(CXX) -o $@ $^ $(TORCH_LINK)

$(OUT)/bin/identity_test: $(OUT)/obj/az/identity_test.o $(COPIES) $(UPSTREAM)
	@mkdir -p $(@D)
	nice -n 19 $(CXX) -o $@ $^ $(TORCH_LINK)

$(addprefix $(OUT)/bin/,$(PLAIN_TESTS) $(PLAIN_EXPERIMENTS)): $(OUT)/bin/%: $(OUT)/obj/plain/%.o
	@mkdir -p $(@D)
	nice -n 19 $(CXX) -o $@ $^ $(LINK)

$(SANITIZED): $(OUT)/bin/sanitized/%: $(OUT)/obj/sanitized/%.o
	@mkdir -p $(@D)
	nice -n 19 $(CXX) $(SANITIZE) -o $@ $^ $(LINK)

# A program must be relinked when libopen_spiel.so (the game) changes.
$(ALL): $(LIB)/libopen_spiel.so

-include $(shell find $(OUT)/obj -name '*.d' 2>/dev/null)
