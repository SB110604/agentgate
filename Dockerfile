FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY examples ./examples
USER app
EXPOSE 8000
CMD ["uvicorn", "examples.procurement:app", "--host", "0.0.0.0", "--port", "8000"]
