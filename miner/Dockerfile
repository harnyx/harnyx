FROM python:3.11-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade uv
RUN useradd --create-home --uid 1000 caster
RUN mkdir -p /sandbox && chown caster:caster /sandbox

WORKDIR /workspace

COPY pyproject.toml uv-workspace.toml mypy.ini .ruff.toml pytest.ini ./
COPY commons ./commons
COPY miner-sdk ./miner-sdk
COPY miner ./miner

RUN chown -R caster:caster /workspace

USER caster

RUN uv sync --package caster-miner

RUN uv tool install ./miner

ENV PATH="/home/caster/.local/bin:${PATH}"

ENTRYPOINT ["/home/caster/.local/bin/caster-miner"]
CMD ["--serve"]
