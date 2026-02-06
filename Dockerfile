FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md requirements.txt /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

CMD ["bash"]
