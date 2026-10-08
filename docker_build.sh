#!/bin/bash
set -e

IMAGE="${IMAGE:-ghcr.io/s-messing/wyoming-qwen3-tts}"
DOCKER="${DOCKER:-$(command -v docker || command -v podman)}"

if ! command -v ruff &>/dev/null || ! command -v mypy &>/dev/null || ! command -v pytest &>/dev/null; then
    echo "Installing dev tools..."
    pip install -q ruff mypy pytest pytest-asyncio
fi

echo "Running lint checks..."
ruff check --fix .
ruff format .

echo "Running type checks..."
mypy wyoming_qwen3_tts/

echo "Running tests..."
pytest -q

if VERSION=$(git describe --tags --exact-match 2>/dev/null); then
    VERSION="${VERSION#v}"  # Strip 'v' prefix
else
    SHA=$(git rev-parse --short HEAD)
    VERSION="0.0.0.dev0+${SHA}"
fi

echo "Building ${IMAGE} with VERSION=${VERSION}"
"$DOCKER" build \
    --build-arg VERSION="${VERSION}" \
    -t "${IMAGE}:latest" .
