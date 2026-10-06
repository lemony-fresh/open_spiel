# Copyright 2026 The Thud-on-OpenSpiel authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Switches a load beside a training run on and off, for load_test_analysis.py.

The load is every az_match process in the idle scheduling class (the evaluation
watcher's matches, watch_run.py; matches at other priorities are left alone). Time is
cut into pairs of `--block` seconds, one block with the load running and one with it
paused (SIGSTOP; SIGCONT resumes it — a match plays the same games however it is
timed), the order within each pair random. Every 5 s it enforces the block's state on
every such process, new ones included, and adds up the CPU time the trainer and the
load used; each block ends as a JSON line in `--log`: pair, load "on" or "off", start
and end (Unix seconds), the trainer's and the load's CPU seconds, the load's processes.
It stops when the trainer has ended or at `--until`, and always resumes the load first.

  python3 thud/experiments/load_toggle.py TRAINER_PID --log FILE [--block 1200]
      [--until "2026-10-06 07:45"] [--seed 1]
"""

import argparse
import datetime
import json
import os
import random
import signal
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")


def load_processes():
  """The az_match processes in the idle scheduling class."""
  found = []
  for name in os.listdir("/proc"):
    if not name.isdigit():
      continue
    try:
      if open(f"/proc/{name}/comm").read().strip() != "az_match":
        continue
      if os.sched_getscheduler(int(name)) == os.SCHED_IDLE:
        found.append(int(name))
    except (OSError, ProcessLookupError):
      continue
  return found


def cpu_seconds(pid):
  """The process's CPU time (user + system, all threads), or None once it is gone."""
  try:
    fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / TICK  # utime, stime: fields 14, 15.
  except (OSError, IndexError):
    return None


def signal_all(sig):
  for pid in load_processes():
    try:
      os.kill(pid, sig)
    except ProcessLookupError:
      pass


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("trainer_pid", type=int)
  parser.add_argument("--log", required=True)
  parser.add_argument("--block", type=float, default=1200)
  parser.add_argument("--until", default=None)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()
  until = (datetime.datetime.strptime(args.until, "%Y-%m-%d %H:%M").timestamp()
           if args.until else float("inf"))
  rng = random.Random(args.seed)

  def stop(signum, frame):
    signal_all(signal.SIGCONT)
    sys.exit(0)
  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)
  signal.signal(signal.SIGHUP, stop)

  try:
    pair = 0
    last = {args.trainer_pid: cpu_seconds(args.trainer_pid)}  # CPU seconds at the last look.
    while True:
      for load in rng.sample(["on", "off"], 2):
        start = time.time()
        trainer_cpu, load_cpu, seen = 0.0, 0.0, set()
        while time.time() < start + args.block:
          signal_all(signal.SIGCONT if load == "on" else signal.SIGSTOP)
          time.sleep(5)
          now = cpu_seconds(args.trainer_pid)
          if now is None or time.time() > until:
            return
          trainer_cpu += now - last[args.trainer_pid]
          last[args.trainer_pid] = now
          for pid in load_processes():
            c = cpu_seconds(pid)
            if c is None:
              continue
            if pid in last:
              load_cpu += c - last[pid]
            last[pid] = c
            seen.add(pid)
        with open(args.log, "a") as f:
          f.write(json.dumps({"pair": pair, "load": load, "start": start, "end": time.time(),
                              "trainer_cpu_s": round(trainer_cpu, 1),
                              "load_cpu_s": round(load_cpu, 1),
                              "load_processes": sorted(seen)}) + "\n")
      pair += 1
  finally:
    signal_all(signal.SIGCONT)


if __name__ == "__main__":
  main()
