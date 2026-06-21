"""OpenAI-compatible inference server for the red-team safety evaluator.

Endpoints
---------
POST /v1/chat/completions
    OpenAI Chat Completions shape. The response being judged is the last
    `assistant` message; the prompt is the preceding `user` message. You may
    instead pass top-level `"prompt"` and `"response"` fields to be explicit.
    The choice message content is a JSON string `{"verdict","reason"}`, and a
    non-standard top-level `"trace"` field holds the full two-stage breakdown.

POST /v1/evaluate
    Explicit `{"prompt": ..., "response": ...}` -> `{verdict, reason, trace}`.

GET  /v1/models
    List the served model id (OpenAI shape).

GET  /health
    Liveness probe.

The `trace` object contains, for each stage (gate, then judge if the gate did
not short-circuit): the parsed verdict/reason, whether it short-circuited, and
the raw LLM messages, completion text, and token usage of that call.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
    # or: python server.py
"""

from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dspy_evaluator import evaluate_safety_traced

MODEL_ID = os.getenv("EVALUATOR_MODEL_ID", "redteam-evaluator-gated")
JUDGE_PATH = os.getenv("DSPY_SAFETY_JUDGE_PATH", "").strip() or "(unoptimized default)"

app = FastAPI(title="Red-Team Safety Evaluator", version="1.0.0")


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] | None = None
    # Convenience overrides — skip the message-role inference entirely.
    prompt: str | None = None
    response: str | None = None
    # Accepted for OpenAI compatibility; streaming is not implemented.
    stream: bool | None = False


class EvaluateRequest(BaseModel):
    prompt: str = ""
    response: str


def _extract_pair(req: ChatCompletionRequest) -> tuple[str, str]:
    """Resolve (prompt, response_to_judge) from the request."""
    if req.response is not None:
        return req.prompt or "", req.response

    messages = req.messages or []
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="Provide either top-level 'response' or a non-empty 'messages' list.",
        )

    # The response under evaluation is the last assistant turn; fall back to the
    # final message when there is no explicit assistant role.
    resp_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "assistant"),
        len(messages) - 1,
    )
    response = messages[resp_idx].content

    prompt = ""
    for i in range(resp_idx - 1, -1, -1):
        if messages[i].role == "user":
            prompt = messages[i].content
            break
    if not prompt:
        prompt = next((m.content for m in messages if m.role == "user"), "")
    return prompt, response


def _aggregate_usage(stages: list[dict]) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for stage in stages:
        for call in stage.get("llm_calls", []):
            usage = (call or {}).get("usage") or {}
            if not isinstance(usage, dict):
                continue
            for key in total:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    total[key] += int(value)
    return total


def _run(prompt: str, response: str) -> dict:
    started = time.time()
    verdict, reason, stages = evaluate_safety_traced(prompt, response)
    latency_ms = round((time.time() - started) * 1000, 2)
    trace = {
        "judge_path": JUDGE_PATH,
        "model": os.getenv("DSPY_MODEL", "gemma-4-31B-it-NVFP4"),
        "stages": stages,
        "final": {"verdict": verdict, "reason": reason},
        "latency_ms": latency_ms,
    }
    return {"verdict": verdict, "reason": reason, "trace": trace}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "judge_path": JUDGE_PATH}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "redteam-evaluator"}],
    }


@app.post("/v1/evaluate")
def evaluate_endpoint(req: EvaluateRequest):
    return _run(req.prompt, req.response)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt, response = _extract_pair(req)
    result = _run(prompt, response)

    content = (
        '{"verdict": "%s", "reason": %s}'
        % (result["verdict"], _json_str(result["reason"]))
    )
    usage = _aggregate_usage(result["trace"]["stages"])
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "trace": result["trace"],
    }


def _json_str(value: str) -> str:
    import json

    return json.dumps(value)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "47923")),
    )
