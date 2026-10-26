FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY app ./app

RUN python -m pip install --upgrade pip && \
    pip install -e .

EXPOSE 7860

CMD ["python", "app/gradio_app.py"]

