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
 && update-ca-certificates --fresh \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
 && python -m pip install --no-cache-dir uv==0.13.5

WORKDIR /workspace

FROM base AS deps

COPY pyproject.toml uv-workspace.toml uv.lock ./
COPY packages/commons ./packages/commons
COPY packages/miner-sdk ./packages/miner-sdk
COPY validator/pyproject.toml ./validator/pyproject.toml

RUN uv sync --package harnyx-validator --frozen --no-install-package harnyx-validator

FROM deps AS runtime

COPY validator ./validator

RUN uv pip install --system --editable ./validator --no-deps

ARG SOURCE_REVISION=""

ENV SOURCE_REVISION="${SOURCE_REVISION}"

ENTRYPOINT ["uv", "run", "--package", "harnyx-validator", "harnyx-validator"]
CMD []
