# CLAUDE.md — Thud on OpenSpiel

This is a **fork of `google-deepmind/open_spiel`**. Our purpose here is to implement the
board game **Thud** as a C++ game inside OpenSpiel, and then build an AI for it.

Our work lives in exactly two places:

- `open_spiel/games/thud/` — the game implementation (`thud.h`, `thud.cc`, `thud_test.cc`)
- `thud/` — our roadmap, session log, rules spec, and experiment scripts

Everything else in this tree is upstream OpenSpiel code.

---

## Session start

Before doing anything else, read:

1. `thud/PROGRESS.md` — the `## Current status` block at the top says where we actually are
   and what the next step is.
2. `thud/PLAN.md` — the roadmap and which phase we are in.
3. `thud/THUD_RULES.md` — the authoritative ruleset and action encoding. **Implement what
   this file says**, not what you remember about Thud. The published rules are ambiguous in
   several places; its section 9 records which reading we chose and why. All of those
   points were settled with the user on 2026-09-22 — if one ever needs changing, raise it
   with the user rather than deviating in code.

## Session end

This project runs across many sessions, so the log is what keeps it coherent. At the end of
each working session, without being asked:

1. Append a dated entry to `thud/PROGRESS.md` under `## Session log` covering: what changed,
   what decisions were made (and why), what is half-finished, and what the next step is.
2. Overwrite the `## Current status` block at the top of `thud/PROGRESS.md` so it reflects
   reality.
3. If the roadmap itself changed — scope, phase order, a deferred decision now resolved —
   update `thud/PLAN.md` too. Do not record roadmap changes only in the session log.

Prefer concrete detail over summary. "Implemented hurl generation; blocked on whether a hurl
onto an empty square is legal" is useful. "Worked on move generation" is not.

---

## Environment

These facts were established by measurement and cost real time to work out. Do not
re-derive them.

- **Development happens in WSL2 / Ubuntu on ARM64.** The host is a Snapdragon X, 10 cores,
  ~31.6 GB RAM — but **WSL itself sees 15 GB RAM + 4 GB swap** (its default is half the
  host). That is ample for the build: full `make -j10` takes ~4m40s with a 1.2 GB peak per
  compiler process; full `ctest -j10` takes ~90 s.
- **Do not develop on the Windows side.** The Windows Python is an x64 build running under
  emulation (its pip tags are `win_amd64`), and no `win_arm64` OpenSpiel wheels exist.
- **Keep the repo on the WSL native filesystem** (`~/thud-openspiel`), never under
  `/mnt/c/...`. Cross-filesystem I/O is drastically slower and makes every build painful.
- **There is no CUDA GPU.** The integrated GPU is a Qualcomm Adreno X1-85. Any serious
  neural-network training will need either cloud compute or a reduced scale.
- **LibTorch for aarch64 comes from PyTorch's pip wheel.** PyTorch publishes no standalone
  LibTorch download for aarch64 Linux, but the wheel `torch==2.10.0` (cp314,
  `manylinux_2_28_aarch64`) ships LibTorch complete — libraries, C++ headers and
  `TorchConfig.cmake` — built with the C++11 ABI, like our clang build (checked
  2026-09-25). OpenSpiel's C++ AlphaZero (`OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON`) builds
  that way, unpatched, in its own folder `build-torch/` (below), and passes its tests and
  the tic-tac-toe control: `thud/PLAN.md` Phase 6.
- **Run LibTorch programs with `OMP_NUM_THREADS=1`.** LibTorch gives every caller 10
  OpenMP threads, and each AlphaZero actor and evaluator calls the network itself, so
  several of them oversubscribe the 10 cores: three searches at once ran at 16
  simulations/s each by default, 128 each with one thread (2026-09-25).
- **The venv has OpenSpiel's pinned JAX set** (`jax==0.9.0.1`, `flax==0.12.3`, ... from
  `open_spiel/scripts/python_extra_deps.sh`), which pins numpy to 2.3.5: flax 0.12.3
  requires numpy below 2.4. The JAX CPU build warns "An NVIDIA GPU may be present"; there
  is none, and it falls back to the CPU as it should.
- OpenSpiel does publish `manylinux_2_28_aarch64` wheels (cp311–cp314). If the C++ build
  breaks, `pip install open_spiel` is a usable fallback to keep Python-side work moving.

## Shell environment — one-time setup, then nothing per session

**Claude Code's Bash tool does not persist shell state between commands.** The working
directory carries over, but exported variables and `source venv/bin/activate` do not. So an
activated venv cannot be relied on across tool calls.

Put the environment in `~/.bashrc` instead, where every new shell inherits it:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/thud-openspiel
export PYTHONPATH=$PYTHONPATH:$HOME/thud-openspiel/build/python
source $HOME/thud-openspiel/venv/bin/activate
```

Do this once. Afterwards there is nothing to do at the start of a session. If a build command
fails with a missing module or a missing `pyspiel`, check `~/.bashrc` first rather than
prefixing activation onto the command.

## Build and test

From the repo root:

```bash
# one-time setup
./install.sh
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt

# build
mkdir -p build && cd build
CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
make -j10

# test
ctest -j10                 # everything
ctest -R thud              # just ours
./examples/example --game=thud
```

`build/` is OpenSpiel's default build type, "Testing" (`-O2`, all runtime checks on). **For
speed measurements use a Release build** (`-O3`, `SPIEL_DCHECK` off) in its own folder, which
git ignores:

```bash
mkdir -p build-release && cd build-release
BUILD_TYPE=Release CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
make -j10 benchmark_game mcts_example   # ~2.5 min; add targets as needed
```

OpenSpiel's C++ AlphaZero needs LibTorch (from the pip wheel, above) and libnop
(`open_spiel/libnop/libnop/`, cloned by `install.sh`; git ignores it). It has its own
Release folder, `build-torch/`, which builds only the C++ targets: `pyspiel` keeps
coming from `build/`, and LibTorch's bundled pybind11 was reported to clash with ours in
a `pyspiel` build (OpenSpiel issue #966; not tried with 2.10):

```bash
mkdir -p build-torch && cd build-torch
BUILD_TYPE=Release OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON OPEN_SPIEL_BUILD_WITH_LIBNOP=ON \
  CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') \
  ../open_spiel
make -j10 torch_integration_test torch_model_test torch_vpnet_test \
  alpha_zero_torch_example alpha_zero_torch_game_example   # ~3.5 min
```

cmake warns "static library kineto_LIBRARY-NOTFOUND not found" (a profiling library
the wheel does not ship); the build does not need it. Our AlphaZero defaults for Thud
are in `thud/experiments/az_thud.flags` (`--flagfile`).

Our own C++ programs that use the AlphaZero model (so far `az_layout_check.cc`) link
against OpenSpiel built as a shared library, as `docs/library.md` describes, in
`build-shared/`, so no upstream CMake file changes: `build-torch/`'s configure plus
`BUILD_SHARED_LIB=ON`, then `make -j10 open_spiel` (~2.5 min). The commands, and the
compiler flags copied from CMake, are in `thud/experiments/build_az_layout_check.sh`.

**Now that JAX and PyTorch are in the venv, the next cmake run in `build/` adds their
Python tests** (OpenSpiel detects both when `OPEN_SPIEL_ENABLE_JAX`/`_PYTORCH` are
unset, `open_spiel/python/CMakeLists.txt:172-190`; `build-torch/`'s configure did), so
`ctest` will then run more than the 285 tests recorded so far.

Playthrough regression baseline (our most valuable correctness tool — it catches accidental
rule changes):

```bash
./open_spiel/scripts/generate_new_playthrough.sh thud
```

Regenerate and diff it after any change to move generation or the action encoding. An
unexpected diff means a rule changed.

---

## Conventions

**Stay out of upstream code.** Do not edit anything under `open_spiel/` except
`open_spiel/games/thud/`, these two registration points:

- `open_spiel/games/CMakeLists.txt` — add the Thud sources and test target
- `open_spiel/python/tests/pyspiel_test.py` — add the `thud` short name

and the generated playthrough baseline `open_spiel/integration_tests/playthroughs/thud.txt`
(a new file, written only by `generate_new_playthrough.sh`, never by hand).

Keeping the diff that small is what lets us rebase onto upstream without pain.

**Branches.** Work on `thud`. Leave `master` tracking upstream so rebasing stays easy.

**Style.** Follow the Google C++ style guide and match the surrounding OpenSpiel code. Mirror
`open_spiel/games/tic_tac_toe/` for the surrounding boilerplate and `open_spiel/games/chess/`
for the from-square × direction × distance action layout. The implementation must be very
fast yet readable — see `thud/PLAN.md` Phase 3.

**Decide from evidence, not preference.** When choosing between approaches — a layout, a
convention, an encoding — check what the upstream project actually documents and what other
implementations actually do, and cite it. "The developer guide documents only this workflow"
settles a question; an unsourced recommendation invites a round trip. This project has
already been bitten the other way: the rules were pinned down by reading three existing Thud
implementations, which overturned two conclusions reached by reasoning alone.

## Licensing — these are obligations, not preferences

OpenSpiel is Apache License 2.0, and we keep Apache 2.0 for our own files too.

- **Every new file** we add carries an Apache 2.0 header with our own copyright line.
- **Every upstream file we modify** must carry a prominent notice stating that we changed it.
  Apache 2.0 section 4(b) requires this. Currently that means `open_spiel/games/CMakeLists.txt`
  and `open_spiel/python/tests/pyspiel_test.py`.
- **Never** remove or alter existing copyright notices, and never touch `LICENSE`.

Note that Thud itself is a commercially published game. Upstreaming our implementation to
OpenSpiel may not be possible — their contributing guide flags that copyrighted games can
need legal approval. Assume this fork is the final home unless that question gets resolved.

**`thud/THUD_RULES.md` quotes the official rules verbatim and must never be upstreamed.**
Game mechanics are not copyrightable but their expression is, so anything that could go
upstream — header comments, a `docs/games.md` entry — must be written from scratch in our own
words. `thud/PLAN.md` has the full pre-PR checklist.

Never upstream `CLAUDE.md` or anything under `thud/`. A `.githooks/pre-push` hook enforces
this for pushes aimed at `deepmind/open_spiel`, but it only runs if this clone has been set
up with `git config core.hooksPath .githooks`, and `--no-verify` bypasses it. **Do not reach
for `--no-verify` if that hook fires** — it is telling you the branch is wrong, not that the
hook is.
