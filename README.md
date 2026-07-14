# vish_agent
SLM-based AI agent, alias "Vish". Shared under

Named for Vishvakarma, Hindu craftsman god.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1

# For GPU acceleration on an NVIDIA card, install a CUDA build of torch first
# (check your driver's max supported CUDA version with `nvidia-smi`):
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install -e .
copy .env.example .env   # then fill in TAVILY_API_KEY
```

## Running tests

```powershell
pytest
```

Tests load the real Qwen2.5-1.5B-Instruct model (no mocking), so the first
run will download the model from Hugging Face and each test session will be
slow to start.

## Architecture

Four modular layers (`src/vish_agent/layers/`): `input_layer.py` ->
`orchestrator.py` -> `tool_executor.py` -> `response_generator.py`, wired
together in `pipeline.py`. Shared infrastructure: `config.py` (env/settings),
`schemas.py` (data contracts between layers), `prompts.py` (prompt
templates), `logging_utils.py` (structured JSONL transaction logging),
`llm/qwen_client.py` (shared model wrapper), and `tools/` (toolbox metadata
+ implementations for IP-API geolocation, NWS weather, and Tavily search).

The pipeline is called through plain Python objects/functions with no
framework coupling, so a REST API layer can be added later without
restructuring existing code.

For fully offline inference after the first run, set
`VISH_MODEL_LOCAL_FILES_ONLY=true` in `.env`.

## Evaluation

`eval_prompts.json` is a standardized, labeled set of prompts (expected
tool + expected arguments per prompt) used to measure tool-selection and
argument-extraction accuracy. `src/vish_agent/evaluation/` grades a pipeline
run against it by exact match; final response correctness isn't
exact-matchable and is left `null` for manual/LLM-judge review.

```powershell
python scripts/run_evaluation.py
```

This reruns the same prompt set against toolboxes of increasing size and
prints tool-selection/argument accuracy per configuration. Full per-prompt
results are logged to `logs/evaluation_results.jsonl`.
