# Contributing to agentgate

Thanks for your interest! Bug reports, docs fixes and features are all welcome.

## Development setup

```bash
git clone https://github.com/SB110604/agentgate.git
cd agentgate
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest            # tests + doctests
ruff check . && ruff format --check .
```

## Pull requests

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Create a branch, keep the change focused, and add or update tests.
3. Make sure `pytest` and `ruff` pass locally — CI runs the same checks on Python 3.10–3.13.
4. Add a line to `CHANGELOG.md` under **Unreleased**.

Look for issues labelled **good first issue** if you are new to the project.

## Code style

Typed Python, `ruff` for lint and formatting (line length 120), small functions, clear errors.
Never commit secrets — configuration comes from environment variables (see `.env.example`).
