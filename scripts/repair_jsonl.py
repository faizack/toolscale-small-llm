"""Remove unparseable lines from results/raw/**/*.jsonl (after a crash / concurrent writers).

A backup of every modified file is written next to it as <name>.jsonl.bak-<timestamp>.
Every removed line is reported (file, line number, first 120 chars) so it can be logged.
Removed calls are simply re-run by the resumable runners.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
total = 0
for p in sorted((ROOT / "results/raw").glob("*/*.jsonl")):
    good, bad = [], []
    with open(p, encoding="utf8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
                good.append(line if line.endswith("\n") else line + "\n")
            except json.JSONDecodeError:
                bad.append((i, line[:120].replace("\n", "\\n")))
    if bad:
        shutil.copy2(p, p.with_name(p.name + f".bak-{stamp}"))
        p.write_text("".join(good), encoding="utf8")
        for i, snippet in bad:
            print(f"removed {p.relative_to(ROOT)}:{i}  {snippet!r}")
        total += len(bad)
print(f"done: {total} corrupt line(s) removed", file=sys.stderr)
