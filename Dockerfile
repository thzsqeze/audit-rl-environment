# Offline verification image: Python 3.11, stdlib-only runtime, pytest for the test suite.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
WORKDIR /app

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Run with: docker run --rm --network none --cpus 2 --memory 4g audit-rl-env
ENTRYPOINT ["python", "-m", "audit_env"]
CMD ["check", "--out", "/app/reports"]
