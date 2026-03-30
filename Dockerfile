FROM python:3.13-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .python-version ./

# Install all third-party dependencies without building the local package.
# The server/ directory is on sys.path via the working directory at runtime.
RUN uv sync --no-dev --no-install-project

# Copy application source
COPY server/ server/
COPY data/ data/
COPY main.py .
COPY __init__.py client.py models.py ./

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
