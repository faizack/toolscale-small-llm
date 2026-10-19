"""Refuse to start a second runner on the same output folder (lesson from 2026-10-18:
two orphaned runners appended to the same JSONL files -> duplicates, a corrupt line, double GPU/RAM load)."""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire(out_dir: Path) -> None:
    lock = out_dir / ".lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid != os.getpid() and _alive(pid):
            sys.exit(f"Another runner (PID {pid}) is writing to {out_dir}. Stop it first.")
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))
