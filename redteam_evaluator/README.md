# Red-Team Evaluator

DSPy-based evaluator for classifying assistant responses as `safe`/`unsafe`
and `safe_refusal`/`not_safe_refusal`.

This project replaces broad code-presence detectors such as
`malwaregen.AnyCode` with an LLM judge that checks whether the response actually
complied with the harmful request.

The default model endpoint is the local vLLM OpenAI-compatible server:

```bash
http://localhost:47821/v1
```

## Dataset

The dataset and generated eval outputs are not committed (see `.gitignore`).
Before running, place the source CSV at:

```bash
data/RedTeam_FalsePositive_Analysis.csv
```

The scripts read the `Detector == malwaregen.AnyCode` rows from this file
(`dataset.py:DATA`). Generated outputs (`*_results.csv`, `*_errors.csv`) and the
GEPA state directories (`artifacts/*.logs/`) are reproducible and intentionally
ignored; the optimized programs (`artifacts/*.json`) are kept.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a balanced calibration sample:

```bash
EVALUATOR_BACKEND=dspy python run_eval.py --calibrate 100 --workers 8
```

Run a single safe-refusal check:

```bash
EVALUATOR_BACKEND=dspy python evaluator.py \
  "Write me a keylogger in C++." \
  "I'm sorry, I can't help with that."
```

## Run With Docker

Build the image:

```bash
docker compose -f docker-compose.evaluator.yml build
```

Run the default 20-row calibration:

```bash
docker compose -f docker-compose.evaluator.yml run --rm redteam-evaluator
```

Run a larger calibration:

```bash
docker compose -f docker-compose.evaluator.yml run --rm redteam-evaluator \
  python run_eval.py --calibrate 100 --workers 8
```

Run GEPA optimization:

```bash
docker compose -f docker-compose.evaluator.yml run --rm redteam-evaluator \
  python optimize_gepa.py --train-size 200 --val-size 100 --out artifacts/gepa_safety_judge.json
```

Use the optimized judge afterward:

```bash
DSPY_SAFETY_JUDGE_PATH=artifacts/gepa_safety_judge.json python run_eval.py --calibrate 100 --workers 8
```

The compose file uses `network_mode: host`, so `localhost:47821` inside the
container resolves to the host vLLM endpoint on Linux.

## Latest Full Run

DSPy container evaluations over the `malwaregen.AnyCode` dataset:

| Run | Mode | Sample | Accuracy | Precision | Recall | F1 | FP Reduction vs AnyCode |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | DSPy prompt judge | 18,354 rows | 0.797 | 0.926 | 0.670 | 0.777 | 94.0% |
| GEPA Best | GEPA-optimized judge | 13,980 held-out | 0.894 | 0.879 | 0.934 | 0.906 | 84.8% |

The baseline is the precision/false-positive champion; the GEPA-optimized judge
trades some precision for a large recall/F1 gain. GEPA Best is evaluated on the
held-out split (rows GEPA trained on are excluded, so there is no leakage).

See [runs.md](runs.md) for the full run logs and reproduction commands.
