FROM python:3.13-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .python-version ./

# Install dependencies (no dev deps, no editable install)
RUN uv sync --no-dev

# Copy application source
COPY server/ server/
COPY data/ data/
COPY main.py .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
