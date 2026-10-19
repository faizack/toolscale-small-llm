# Setup guide

This guide covers the prerequisites and steps to prepare the project, run its experiments, analyze the results, and build the paper.

## Project and hardware

The project studies tool-selection accuracy and failure modes in small language-model agents, using Ollama for local inference.

The documented runs were made on Windows 11 with an RTX 3050 Laptop GPU (4 GB VRAM) and 16 GB RAM. The core and extra experiment scripts take approximately four hours in total on that machine. Runtime and memory requirements vary by hardware; the 16k-context multi-turn runs can be memory-intensive.

## Prerequisites

Install the following before continuing:

- **Python 3.12**
- **uv**, for creating the virtual environment and installing the pinned dependencies
- **Ollama 0.32 or newer**, with enough disk space for the model weights
- **Bash** to run the experiment scripts. On Windows, use Git Bash; Linux and macOS can use their system Bash.
- **MiKTeX or TeX Live with `latexmk`**, only if you want to compile the paper

Run the commands below from the repository root. In Windows Git Bash, the virtual environment's Python path is `.venv/Scripts/python.exe`; on Linux and macOS it is `.venv/bin/python`.

## 1. Create the Python environment

```bash
uv venv --python 3.12 .venv
```

Install the exact versions pinned in the lock file:

```bash
# Windows
uv pip install --python .venv/Scripts/python.exe --no-deps -r requirements.lock.txt

# Linux or macOS
uv pip install --python .venv/bin/python --no-deps -r requirements.lock.txt
```

Keep `--no-deps`: `bfcl-eval` has optional dependencies that would otherwise pull in PyTorch (about 5 GB). This project uses its pure-Python components.

## 2. Check the data

The BFCL v4 and ToolRet data used by the experiments are included under `data/raw/`. Verify their checksums from the repository root:

```bash
sha256sum -c data/raw/SHA256SUMS.txt
```

If `sha256sum` is unavailable, compare each listed file's SHA-256 hash in `data/raw/SHA256SUMS.txt` using an equivalent tool. Do not edit the data files unless you also intend to update and verify the checksums.

## 3. Install and check the Ollama models

Start Ollama and pull the models used by the full set of experiments:

```bash
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:3b-instruct
ollama pull llama3.2
ollama pull qwen3.5:2b
ollama pull qwen3-embedding:0.6b
ollama pull bge-m3
```

`llama3.2` is the Ollama tag used for Llama-3.2-3B-Instruct. The embedding models are needed for retrieval experiments. To check that Ollama is running and the models are available, run:

```bash
ollama list
```

On Windows, a low-memory Ollama server configuration is available:

```powershell
.\scripts\start_ollama_lowmem.ps1
```

Run this in a separate PowerShell window and leave it open while experiments run. It limits Ollama to one parallel request and one loaded model.

## 4. Run the experiments

In Git Bash on Windows, or Bash on Linux/macOS, run from the repository root:

```bash
# Core runs: RQ1-RQ3
bash scripts/run_all.sh

# Extra runs: RQ4, RQ5, and ToolRet retrieval
bash scripts/run_extra.sh
```

The scripts use the existing `.venv` and are resumable: rerun a script after an interruption. The extra runs use additional data and are independent of the core runs, but the usual full workflow runs the core experiments first.

For an initial smoke test, run only three examples from the RQ1 scaling configuration:

```bash
# Windows
.venv/Scripts/python.exe scripts/run_experiment.py configs/rq1_scaling.yaml --limit 3

# Linux or macOS
.venv/bin/python scripts/run_experiment.py configs/rq1_scaling.yaml --limit 3
```

Each experiment's settings are in its corresponding file in `configs/`. Core results cover scaling, failure categories, and retrieval; extra results cover native BFCL tool lists, multi-turn BFCL tasks, and ToolRet.

## 5. Analyze results

After the runs finish, regenerate the processed results, tables, figures, and paper macros:

```bash
# Windows
.venv/Scripts/python.exe scripts/analyze.py
.venv/Scripts/python.exe scripts/analyze_extra.py
.venv/Scripts/python.exe scripts/make_macros.py

# Linux or macOS
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/analyze_extra.py
.venv/bin/python scripts/make_macros.py
```

The processed results are written under `results/processed/`; generated paper tables and figures are under `paper/tables/` and `paper/figures/`.

## 6. Build the paper (optional)

With MiKTeX or TeX Live and `latexmk` installed, run from the repository root:

```bash
cd paper
latexmk
```

The compiled PDF is `paper/build/main.pdf`. Build output is kept in `paper/build/`.

## Reproducibility notes

- The experiment logs include model/runtime details; model digests can differ if Ollama changes a tag.
- Temperature-zero generation is not guaranteed to be bitwise reproducible when the KV cache is reused.
- Ollama may drop malformed or unknown-tool calls. The experiment code captures raw output to help classify these cases.
- `qwen3.5:2b` uses an 8-bit model in the documented setup; the other listed language models use 4-bit quantization.
- To understand the research questions, recorded caveats, and evidence behind the results, see the [README](README.md) and the lab notebook in `notes/`.
