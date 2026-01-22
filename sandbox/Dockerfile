FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 caster
RUN mkdir -p /sandbox && chown caster:caster /sandbox

WORKDIR /workspace

COPY packages/miner-sdk ./packages/miner-sdk
COPY sandbox ./sandbox

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir ./packages/miner-sdk ./sandbox

USER caster

ENTRYPOINT ["caster-sandbox"]
CMD ["--serve"]

