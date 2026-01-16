FROM python:3.11-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg build-essential \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && printf "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian %s stable\n" "$(. /etc/os-release && echo "$VERSION_CODENAME")" \
    > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --upgrade uv

WORKDIR /workspace

FROM base AS deps

COPY pyproject.toml uv-workspace.toml uv.lock ./
COPY commons ./commons
COPY miner-sdk ./miner-sdk
COPY validator/pyproject.toml ./validator/pyproject.toml

RUN uv sync --package caster-validator --frozen --no-install-package caster-validator

FROM deps AS runtime

COPY validator ./validator

RUN uv pip install --system --editable ./validator --no-deps

ENTRYPOINT ["caster-validator"]
CMD []
