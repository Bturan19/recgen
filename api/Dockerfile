FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY recgen/ recgen/
COPY api/ api/

RUN pip install --no-cache-dir fastapi uvicorn && \
    pip install --no-cache-dir .

ENV RECGEN_MODEL_DIR=/models RECGEN_DEVICE=cpu RECGEN_POOLING=mean

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
