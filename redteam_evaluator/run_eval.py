"""Run the LLM evaluator over the red-team dataset and report metrics.

Replaces the `malwaregen.AnyCode` detector verdict with our evaluator's
verdict and compares both against the human ground-truth ("Is it actually
safe?").

Usage:
  python run_eval.py --calibrate 200   # balanced sample, report accuracy
  python run_eval.py --full            # all malwaregen.AnyCode rows -> CSV
"""

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Re-exported from dataset so the loader/split/ground-truth logic is shared with
# optimize_gepa.py (keeps the GEPA train/val split and eval in lock-step).
from dataset import (  # noqa: F401
    DATA,
    DETECTOR,
    load_rows,
    row_key,
    train_val_keys,
    truth_is_safe,
)
from evaluator import evaluate

csv.field_size_limit(10**9)


def eval_row(row):
    try:
        verdict, reason = evaluate(row["Prompt"], row["Output"])
    except Exception as e:  # noqa: BLE001
        verdict, reason = "unsafe", f"ERROR: {e}"  # fail-safe
    return row, verdict, reason


def run(rows, workers, progress_every=50):
    results = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(eval_row, r) for r in rows]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % progress_every == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                print(
                    f"  {done}/{len(rows)}  ({rate:.1f}/s)",
                    file=sys.stderr,
                    flush=True,
                )
    return results


def metrics(results):
    """Treat UNSAFE as the positive (flagged) class."""
    tp = fp = tn = fn = 0
    for row, verdict, _ in results:
        pred_unsafe = verdict == "unsafe"
        actual_unsafe = not truth_is_safe(row)
        if pred_unsafe and actual_unsafe:
            tp += 1
        elif pred_unsafe and not actual_unsafe:
            fp += 1
        elif not pred_unsafe and not actual_unsafe:
            tn += 1
        else:
            fn += 1
    n = len(results)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(n=n, tp=tp, fp=fp, tn=tn, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)


def print_metrics(m):
    print(f"\n  N={m['n']}")
    print("  Confusion (positive = UNSAFE / flagged):")
    print(f"    TP (unsafe->unsafe): {m['tp']:5d}   FN (unsafe->safe): {m['fn']:5d}")
    print(f"    FP (safe->unsafe):   {m['fp']:5d}   TN (safe->safe):   {m['tn']:5d}")
    print(f"  Accuracy : {m['acc']:.3f}")
    print(f"  Precision: {m['prec']:.3f}   Recall: {m['rec']:.3f}   F1: {m['f1']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, default=0,
                    help="balanced sample size (half safe, half unsafe)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/evaluator_results.csv")
    ap.add_argument("--errors", default=None, help="write misclassified rows here")
    ap.add_argument(
        "--exclude-gepa-train",
        action="store_true",
        help="drop the rows GEPA trained/validated on before evaluating, so "
        "the optimized judge is scored only on held-out data (no leakage)",
    )
    ap.add_argument("--gepa-train-size", type=int, default=200)
    ap.add_argument("--gepa-val-size", type=int, default=100)
    ap.add_argument("--gepa-seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rows()
    print(f"Loaded {len(rows)} {DETECTOR} rows", file=sys.stderr)

    if args.exclude_gepa_train:
        excluded = train_val_keys(
            rows, args.gepa_train_size, args.gepa_val_size, args.gepa_seed
        )
        before = len(rows)
        rows = [r for r in rows if row_key(r) not in excluded]
        print(
            f"Excluded {before - len(rows)} GEPA train/val rows "
            f"(train={args.gepa_train_size} val={args.gepa_val_size} "
            f"seed={args.gepa_seed}); {len(rows)} held-out rows remain",
            file=sys.stderr,
        )

    if args.calibrate:
        rng = random.Random(args.seed)
        safe = [r for r in rows if truth_is_safe(r)]
        unsafe = [r for r in rows if not truth_is_safe(r)]
        k = args.calibrate // 2
        sample = rng.sample(safe, min(k, len(safe))) + rng.sample(unsafe, min(k, len(unsafe)))
        rng.shuffle(sample)
        rows = sample
        print(f"Calibrating on {len(rows)} balanced rows", file=sys.stderr)

    results = run(rows, args.workers)
    m = metrics(results)
    print_metrics(m)

    # Baseline: malwaregen.AnyCode flags everything as unsafe.
    base_fp = sum(1 for r in rows if truth_is_safe(r))
    print(f"\n  Baseline malwaregen.AnyCode: flags all {len(rows)} as UNSAFE")
    print(f"    -> {base_fp} false positives. Evaluator FP: {m['fp']} "
          f"(reduced {base_fp - m['fp']}, {100*(base_fp-m['fp'])/base_fp:.1f}%)")

    if args.errors:
        with open(args.errors, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Prompt", "Output", "truth_safe", "pred", "pred_reason"])
            for row, verdict, reason in results:
                if (verdict == "unsafe") != (not truth_is_safe(row)):
                    w.writerow([row["Prompt"][:500], row["Output"][:800],
                                truth_is_safe(row), verdict, reason])
        print(f"\n  Wrote misclassifications to {args.errors}", file=sys.stderr)

    if args.full:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Prompt", "Output", "Is it actually safe?",
                        "evaluator_verdict", "evaluator_reason", "Detector"])
            for row, verdict, reason in results:
                w.writerow([row["Prompt"], row["Output"], row["Is it actually safe?"],
                            verdict, reason, "evaluator.LLMJudge"])
        print(f"\n  Wrote {len(results)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
