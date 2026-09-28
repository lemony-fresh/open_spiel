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

"""Copies OpenSpiel's C++ AlphaZero and its MCTS into thud/az/ (thud/PLAN.md Phase 6,
roadmap stage 2), changed only mechanically, so the copy can be changed without touching
upstream code and linked next to libopen_spiel.so without clashing with the originals:

- namespace open_spiel::algorithms becomes open_spiel::thud_az (so torch_az becomes
  open_spiel::thud_az::torch_az);
- includes of the copied files point at the copies;
- header guards are renamed;
- each file gets a notice that it was changed (Apache License 2.0, section 4(b)).

Everything else, parameter names included, stays byte for byte as upstream, so networks
and checkpoints stay interchangeable. Behavioural changes come later, as separate commits.

  python3 thud/az/import_from_upstream.py            # overwrites the copies in thud/az/
  python3 thud/az/import_from_upstream.py --check    # compares them, changes nothing

--check regenerates every copy in memory and compares it with thud/az/: before any later
change it proves the copies are upstream's code with only the renames above; after
changes it lists exactly the files that differ from a fresh import, and exits non-zero.
"""

import os
import subprocess
import sys

FILES = {  # Upstream path (under open_spiel/) -> name in thud/az/.
    "algorithms/mcts.h": "mcts.h",
    "algorithms/mcts.cc": "mcts.cc",
    "algorithms/alpha_zero_torch/alpha_zero.h": "alpha_zero.h",
    "algorithms/alpha_zero_torch/alpha_zero.cc": "alpha_zero.cc",
    "algorithms/alpha_zero_torch/device_manager.h": "device_manager.h",
    "algorithms/alpha_zero_torch/model.h": "model.h",
    "algorithms/alpha_zero_torch/model.cc": "model.cc",
    "algorithms/alpha_zero_torch/vpevaluator.h": "vpevaluator.h",
    "algorithms/alpha_zero_torch/vpevaluator.cc": "vpevaluator.cc",
    "algorithms/alpha_zero_torch/vpnet.h": "vpnet.h",
    "algorithms/alpha_zero_torch/vpnet.cc": "vpnet.cc",
    "examples/alpha_zero_torch_example.cc": "az_trainer.cc",
}

REPLACEMENTS = [  # In order.
    ('#include "open_spiel/algorithms/mcts.h"', '#include "thud/az/mcts.h"'),
    ('#include "open_spiel/algorithms/alpha_zero_torch/', '#include "thud/az/'),
    ("OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_", "THUD_AZ_"),
    ("OPEN_SPIEL_ALGORITHMS_MCTS_H_", "THUD_AZ_MCTS_H_"),
    ("open_spiel::algorithms::", "open_spiel::thud_az::"),
    ("\nnamespace algorithms {\n", "\nnamespace thud_az {\n"),
    ("}  // namespace algorithms\n", "}  // namespace thud_az\n"),
]

LICENSE_END = "// limitations under the License.\n"


def main():
    check = sys.argv[1:] == ["--check"]
    differ = []
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%h (%ad)", "--date=short", "--",
         *("open_spiel/" + path for path in FILES)],
        cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    for upstream, name in FILES.items():
        with open(os.path.join(repo, "open_spiel", upstream)) as f:
            text = f.read()
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        notice = (
            "\n// Changed by the Thud-on-OpenSpiel authors, 2026: copied from\n"
            f"// open_spiel/{upstream} at upstream commit {commit}\n"
            "// by thud/az/import_from_upstream.py, which renames the namespace\n"
            "// open_spiel::algorithms to open_spiel::thud_az, points the includes at\n"
            "// the copies and renames the header guard. Later changes: git history\n"
            "// and thud/PLAN.md, Phase 6.\n")
        # Nothing may still refer to the upstream copies (checked before the notice,
        # which names the upstream file).
        assert "namespace algorithms" not in text, upstream
        assert "open_spiel/algorithms/mcts.h" not in text, upstream
        assert "alpha_zero_torch/" not in text, upstream
        assert text.count(LICENSE_END) == 1, upstream
        text = text.replace(LICENSE_END, LICENSE_END + notice)
        target = os.path.join(repo, "thud", "az", name)
        if check:
            with open(target) as f:
                same = f.read() == text
            print(f"{'same    ' if same else 'DIFFERS '} thud/az/{name} <- open_spiel/{upstream}")
            if not same:
                differ.append(name)
            continue
        with open(target, "w") as f:
            f.write(text)
        print(f"thud/az/{name} <- open_spiel/{upstream}")
    if check:
        print(f"{len(FILES) - len(differ)} of {len(FILES)} copies are exactly a fresh import"
              + (f"; differing: {', '.join(differ)}" if differ else ""))
        sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
