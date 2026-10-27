FROM python:3.12.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable && useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "inference_lab.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
