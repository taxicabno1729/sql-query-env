# Entry point for openenv validator and uv run.
# Re-exports the FastAPI app and provides a main() launcher.
import uvicorn
from server.main import app  # noqa: F401


def main() -> None:
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
