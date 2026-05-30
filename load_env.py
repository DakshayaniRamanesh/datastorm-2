"""Load project-root .env into os.environ (optional python-dotenv)."""

from pathlib import Path


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass
