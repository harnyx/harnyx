#!/usr/bin/env bash
set -euo pipefail

validator_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${validator_dir}/docker-compose.yml"

docker compose --project-directory "${validator_dir}" -f "${compose_file}" logs -f --tail 200 "$@"
