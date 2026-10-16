#!/usr/bin/env bash
set -e

# GUI (X11) 許可
xhost +local:root >/dev/null 2>&1 || true

# Docker Compose でコンテナ起動・アタッチ
docker compose run --rm omx-dev /bin/bash
