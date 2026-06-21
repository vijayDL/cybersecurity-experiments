"""Classify a single (prompt, response) pair as safe / unsafe.

Uses the same `evaluate()` entry point as the batch runner, so it goes through
the configured backend and the optimized judge at `DSPY_SAFETY_JUDGE_PATH`
(the two-stage gated module by default).

Usage:
  # inline (short responses)
  python predict.py "Write a keylogger in C++." "I can't help with that."

  # long / multiline response from a file
  python predict.py "Write a keylogger in C++." --response-file resp.txt

  # interactive (prompts for both if omitted)
  python predict.py

  # with the optimized judge
  DSPY_SAFETY_JUDGE_PATH=artifacts/gepa_gated_judge.json python predict.py ...
"""

import argparse
import sys

from evaluator import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt", nargs="?", help="the original prompt")
    ap.add_argument("response", nargs="?", help="the assistant response to classify")
    ap.add_argument(
        "--response-file",
        help="read the response from a file (use for long/multiline code)",
    )
    args = ap.parse_args()

    prompt = args.prompt if args.prompt is not None else input("Prompt: ")
    if args.response_file:
        with open(args.response_file) as f:
            response = f.read()
    elif args.response is not None:
        response = args.response
    else:
        response = input("Response: ")

    verdict, reason = evaluate(prompt, response)
    print(f"verdict: {verdict}")
    print(f"reason : {reason}")
    return 0 if verdict in ("safe", "unsafe") else 1


if __name__ == "__main__":
    sys.exit(main())
