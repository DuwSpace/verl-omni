#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-verl-omni:npu-a2-omnift}"
IMAGE_FILE="${IMAGE_FILE:-}"

if [[ -n "${IMAGE_FILE}" ]]; then
  if [[ ! -f "${IMAGE_FILE}" ]]; then
    echo "error: Docker image file does not exist: ${IMAGE_FILE}" >&2
    exit 1
  fi

  case "${IMAGE_FILE}" in
    *.tar.gz|*.tgz)
      gzip -dc "${IMAGE_FILE}" | docker load
      ;;
    *.tar.zst|*.tzst)
      zstd -dc "${IMAGE_FILE}" | docker load
      ;;
    *)
      docker load --input "${IMAGE_FILE}"
      ;;
  esac
else
  docker build \
    --file "${REPO_ROOT}/docker/Dockerfile.a2.npu" \
    --tag "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "error: image ${IMAGE_NAME} was not found after setup" >&2
  echo "Set IMAGE_NAME to the tag contained in IMAGE_FILE, or build without IMAGE_FILE." >&2
  exit 1
fi

docker image inspect "${IMAGE_NAME}" --format 'ready: {{.RepoTags}} {{.Id}}'
