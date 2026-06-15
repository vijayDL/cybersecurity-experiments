FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DSPY_API_BASE=http://localhost:47821/v1 \
    DSPY_MODEL=gemma-4-31B-it-NVFP4 \
    EVALUATOR_BACKEND=dspy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_eval.py", "--calibrate", "20", "--workers", "4"]
