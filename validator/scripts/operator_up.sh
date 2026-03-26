#!/usr/bin/env bash
set -euo pipefail

validator_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${validator_dir}/docker-compose.yml"

WATCHTOWER_DOCKER_API_VERSION="${WATCHTOWER_DOCKER_API_VERSION:-1.44}" \
docker compose --project-directory "${validator_dir}" -f "${compose_file}" up -d --pull always
