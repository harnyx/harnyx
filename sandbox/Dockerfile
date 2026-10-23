FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 harnyx
RUN mkdir -p /sandbox && chown harnyx:harnyx /sandbox

WORKDIR /workspace

COPY packages/miner-sdk ./packages/miner-sdk
COPY sandbox ./sandbox

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && update-ca-certificates --fresh \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
  && pip install --no-cache-dir "certifi>=2026.10.19" ./packages/miner-sdk ./sandbox

ARG SOURCE_REVISION=""

ENV SOURCE_REVISION="${SOURCE_REVISION}"

USER harnyx

ENTRYPOINT ["harnyx-sandbox"]
CMD ["--serve"]
