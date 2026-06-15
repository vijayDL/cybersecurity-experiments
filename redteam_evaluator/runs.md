# Evaluation Runs

This file records the two reference evaluator runs: the un-optimized DSPy
prompt judge (**Baseline**) and the GEPA-optimized judge (**GEPA Best**).

## Baseline: DSPy Prompt Judge, Full Dataset

Status: completed

Command:

```bash
docker compose -f docker-compose.evaluator.yml run --rm redteam-evaluator \
  python run_eval.py --workers 16 --full \
    --out data/dspy_full_results.csv \
    --errors data/dspy_full_errors.csv
```

Environment:

| Field | Value |
|---|---|
| Evaluator backend | `dspy` (un-optimized prompt judge) |
| Endpoint | `http://localhost:47821/v1` |
| Model | `gemma-4-31B-it-NVFP4` |
| Container image | `redteam-evaluator:dspy` |
| Dataset filter | `Detector == malwaregen.AnyCode` |
| Sample | Full filtered dataset |
| Sample size | 18,354 rows |
| Workers | 16 |
| Full predictions | `data/dspy_full_results.csv` |
| Misclassifications | `data/dspy_full_errors.csv` |

Metrics:

| Metric | Value |
|---|---:|
| N | 18,354 |
| TP | 6,512 |
| FN | 3,208 |
| FP | 520 |
| TN | 8,114 |
| Accuracy | 0.797 |
| Precision | 0.926 |
| Recall | 0.670 |
| F1 | 0.777 |
| Baseline AnyCode false positives | 8,634 |
| Evaluator false positives | 520 |
| False-positive reduction | 8,114 / 8,634, 94.0% |

Confusion matrix:

| Actual / Predicted | Unsafe | Safe |
|---|---:|---:|
| Unsafe | 6,512 | 3,208 |
| Safe | 520 | 8,114 |

Notes:

- `UNSAFE` is the positive class (the `run_eval.py` convention).
- Precision/false-positive champion: it rarely over-flags (94% fewer false
  positives than AnyCode), but misses ~33% of genuinely-unsafe responses
  (recall 0.670).

## Run 2: GEPA Optimization Loop

Status: completed

Command:

```bash
docker compose -f docker-compose.evaluator.yml run --rm redteam-evaluator \
  python optimize_gepa.py --train-size 4000 --val-size 300 --out artifacts/gepa_safety_judge_best.json --workers 8
```

Evaluation command (full held-out set; excludes the exact rows GEPA trained on, so there is no train/test leakage):

```bash
docker compose -f docker-compose.evaluator.yml run --rm \
  -e DSPY_SAFETY_JUDGE_PATH=artifacts/gepa_safety_judge_best.json \
  redteam-evaluator python run_eval.py --full --workers 8 \
    --exclude-gepa-train --gepa-train-size 4000 --gepa-val-size 300 --gepa-seed 42 \
    --out data/best_heldout_full.csv --errors data/best_heldout_errors.csv
```

Environment:

| Field | Value |
|---|---|
| Evaluator backend | `dspy` |
| Optimized program | `artifacts/gepa_safety_judge_best.json` |
| Endpoint | `http://localhost:47821/v1` |
| Model | `gemma-4-31B-it-NVFP4` |
| GEPA train / val | 4,000 / 300 (balanced, seed 42) |
| GEPA budget | `--auto heavy` (3,135 rollouts; converged ~iter 7 of 22) |
| GEPA metric | correct = 1.0, false negative = 0.0, false positive = 0.0 |
| Dataset filter | `Detector == malwaregen.AnyCode` |
| Sample | Full dataset minus GEPA train/val rows |
| Held-out size | 13,980 rows (6,411 safe / 7,569 unsafe) |
| Workers | 8 |
| Full predictions | `data/best_heldout_full.csv` |
| Misclassifications | `data/best_heldout_errors.csv` |

Metrics:

| Metric | Value |
|---|---:|
| N | 13,980 |
| TP | 7,067 |
| FN | 502 |
| FP | 973 |
| TN | 5,438 |
| Accuracy | 0.894 |
| Precision | 0.879 |
| Recall | 0.934 |
| F1 | 0.906 |
| Baseline AnyCode false positives | 6,411 |
| Evaluator false positives | 973 |
| False-positive reduction | 5,438 / 6,411, 84.8% |

Confusion matrix:

| Actual / Predicted | Unsafe | Safe |
|---|---:|---:|
| Unsafe | 7,067 | 502 |
| Safe | 973 | 5,438 |

Notes:

- Recall / F1 champion: catches 93% of genuinely-unsafe responses (vs 67% for
  the baseline), lifting F1 from 0.777 to 0.906, at the cost of some precision
  (0.926 -> 0.879) and a higher false-positive rate (FP reduction 94.0% ->
  84.8%).
- Convergence mattered more than the FP penalty value: a light GEPA run with
  FP=0.0 stalled at recall 0.810 / F1 0.827; the converged heavy run recovered
  to recall 0.934 / F1 0.906 under the same metric.
- Metrics are leakage-free: the eval excludes the 4,374 rows (4,289 unique
  keys) GEPA trained/validated on. Comparison caveat: the Baseline above is
  measured on the full 18,354 rows while GEPA Best is on the 13,980 held-out
  rows; the un-optimized baseline judge trains on nothing, so its per-row rates
  are not materially affected by the difference.
- **Evaluation Discrepancy Note**: The file `logs/full_best_heldout.log` records a parallel evaluation run of the same judge on `N = 14,690` held-out rows (Accuracy 0.894, Precision 0.878, Recall 0.933, F1 0.905). This run mistakenly used `--gepa-train-size 3000 --gepa-val-size 600` for exclusion, which under-excluded 1,000 training rows and introduced minor data leakage. The metrics in the table above (on 13,980 rows) are the correct, leak-free evaluations where train/val split sizes match.
- The 973 false positives are recorded in `data/best_heldout_errors.csv` and are
  the primary lever for further precision gains.
