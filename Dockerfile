FROM python:3.13-slim

WORKDIR /app

# Install only the packages the server needs to run (no openenv-core/gradio/numpy)
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    "uvicorn[standard]>=0.34" \
    pydantic>=2.10

# Copy application source
COPY server/ server/
COPY data/ data/
COPY main.py .
COPY __init__.py client.py models.py ./

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
