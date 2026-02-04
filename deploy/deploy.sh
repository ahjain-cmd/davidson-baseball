#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/opt/davidson_baseball"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repo not found at $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"

echo "Pulling latest code..."
git pull --ff-only

echo "Rebuilding and restarting containers..."
docker compose --env-file "$REPO_DIR/.env" up -d --build

echo "Done. App should be live on :8501"
