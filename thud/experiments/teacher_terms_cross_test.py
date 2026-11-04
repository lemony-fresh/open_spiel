#!/usr/bin/env python3
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

"""The teacher's positional terms in C++ (thud/az/teacher_terms.h) against the Python fit's
(thud/experiments/teacher_terms.py), which set their weights: 400 positions of random games,
uniform and capture-greedy, written by `teacher_terms_test dump`, each term equal in both.
The control, required: positions with a hurl reach above 0 and with several troll-pair
counts among them (else the comparison would prove little).

  python3 thud/experiments/teacher_terms_cross_test.py BUILD_TESTS_BIN
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_terms  # noqa: E402

binary = os.path.join(sys.argv[1], "teacher_terms_test")
out = subprocess.run([binary, "dump", "400", "11"], check=True, capture_output=True,
                     text=True).stdout
wrong, reaching, pair_counts, n = 0, 0, set(), 0
for line in out.splitlines():
  board, reach, pairs = line.split("\t")
  python = teacher_terms.terms(teacher_terms.parse(board.replace("|", "\n")))
  n += 1
  if (python["dwarf_hurl_reach"], python["troll_cohesion"]) != (int(reach), int(pairs)):
    wrong += 1
    if wrong <= 3:
      print(f"differs: C++ reach {reach} pairs {pairs}, Python {python['dwarf_hurl_reach']} "
            f"{python['troll_cohesion']} in\n{board.replace('|', chr(10))}")
  reaching += python["dwarf_hurl_reach"] > 0
  pair_counts.add(python["troll_cohesion"])
ok = n == 400 and wrong == 0 and reaching >= 50 and len(pair_counts) >= 4
print(f"{n} positions, {wrong} differing; {reaching} with a hurl reach, troll pair counts "
      f"{sorted(pair_counts)}")
print("teacher_terms_cross_test: " + ("all passed" if ok else "FAILED"))
sys.exit(0 if ok else 1)
