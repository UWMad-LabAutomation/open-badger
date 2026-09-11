#!/usr/bin/env bash

# Build and run the Open Badger development container.
# Usage:
#   ./container.sh build
#   ./container.sh run
#   ./container.sh run python3 scripts/train.py

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${OPEN_BADGER_IMAGE:-open-badger:dev}"
CONTAINER_NAME="${OPEN_BADGER_CONTAINER:-open-badger-dev}"

usage() {
    echo "Usage: $0 {build|run} [command...]" >&2
    echo "Environment: OPEN_BADGER_IMAGE, OPEN_BADGER_CONTAINER" >&2
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

command="$1"
shift

case "$command" in
    build)
        # Build the image from the repository root so Docker can access the
        # package metadata, source tree, configs, and scripts.
        docker build \
            --file "$ROOT_DIR/tools/docker/Dockerfile" \
            --tag "$IMAGE_NAME" \
            "$ROOT_DIR"
        ;;

    run)
        if [[ $# -eq 0 ]]; then
            set -- bash
        fi

        docker run --rm -it \
            --name "$CONTAINER_NAME" \
            --runtime=nvidia \
            -e NVIDIA_VISIBLE_DEVICES=all \
            --volume "$ROOT_DIR:/workspace/open-badger" \
            --workdir /workspace/open-badger \
            "$IMAGE_NAME" \
            "$@"
        ;;

    *)
        usage
        exit 2
        ;;
esac
