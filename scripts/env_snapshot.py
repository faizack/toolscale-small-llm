"""Record the exact environment (for reproducibility) to logs/env_<timestamp>.json."""
import json, platform, subprocess, sys
from datetime import datetime
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]

def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"

snap = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "git_commit": sh(["git", "rev-parse", "HEAD"]),
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    "ollama_version": httpx.get("http://127.0.0.1:11434/api/version").json()["version"],
    "ollama_models": [{"name": m["name"], "digest": m["digest"], "size": m["size"],
                       "quant": m["details"].get("quantization_level"),
                       "params": m["details"].get("parameter_size")}
                      for m in httpx.get("http://127.0.0.1:11434/api/tags").json()["models"]],
    "bfcl_data_commit": "ShishirPatil/gorilla@f7cf7359b7ac615a0b294831c5ba2bc95ee4a000",
    "pip_freeze": sh([sys.executable, "-m", "pip", "freeze"]).splitlines()
                  if "unavailable" not in sh([sys.executable, "-m", "pip", "--version"]) else
                  (ROOT / "requirements.lock.txt").read_text().splitlines(),
}
(ROOT / "logs").mkdir(exist_ok=True)
out = ROOT / "logs" / f"env_{datetime.now():%Y%m%d-%H%M%S}.json"
out.write_text(json.dumps(snap, indent=2))
print("wrote", out)
