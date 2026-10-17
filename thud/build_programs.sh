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

# Brings the programs our runs use up to date (thud/programs.mk), so that no run starts
# with a stale one: the game's library first (build-shared/), then every program whose
# sources changed. With nothing to do it takes a second. Every queue in ~/thud-runs runs
# it before starting (CLAUDE.md). It refuses to replace a program while one of ours from
# build-shared/ is running — a run's next segment would start the new binary — and says
# which; it then exits 2 without building (1: a build failed).
#
#   thud/build_programs.sh [PROGRAM ...]

set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
export PATH="$repo/venv/bin:$PATH"

# The library is current when no source of the game (ours; upstream's change only with a
# rebase) is newer than it — CMake's Makefiles always answer `make -q` with "out of date".
game_changed=$(find open_spiel/games/thud -newer build-shared/libopen_spiel.so -type f | head -1)
if [ -z "$game_changed" ] && make -f thud/programs.mk -q "$@" > /dev/null 2>&1; then
  echo "build_programs: up to date"
  exit 0
fi
running=$(ps -eo pid,args | awk '$2 ~ /build-shared\/[^/ ]+$/ {print}')
if [ -n "$running" ]; then
  echo "build_programs: programs of ours are running, so nothing is replaced:" >&2
  echo "$running" | cut -c1-150 >&2
  exit 2
fi
available_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
jobs=$(( available_mb / 1536 ))
[ "$jobs" -gt 10 ] && jobs=10
[ "$jobs" -lt 1 ] && jobs=1
echo "build_programs: building ($jobs jobs) ..."
nice -n 19 make -C build-shared -j"$jobs" -s open_spiel > /dev/null &&
  make -f thud/programs.mk -j"$jobs" -s "$@" || { echo "build_programs: FAILED" >&2; exit 1; }
echo "build_programs: done"
