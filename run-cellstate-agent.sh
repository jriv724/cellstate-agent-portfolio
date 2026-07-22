#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CELLSTATE_PYTHON_BIN:-python}"
ENV_FILE="${CELLSTATE_ENV_FILE:-}"

if [[ -z "$ENV_FILE" ]]; then
    if [[ -f "$REPO_DIR/.env.cellstate.local" ]]; then
        ENV_FILE="$REPO_DIR/.env.cellstate.local"
    elif [[ -f "$REPO_DIR/config/resources.env" ]]; then
        ENV_FILE="$REPO_DIR/config/resources.env"
    fi
fi
if [[ -n "$ENV_FILE" ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "CellState environment file not found: $ENV_FILE" >&2
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${CELLSTATE_ATLAS_PATH:-}" ]]; then
    echo "CELLSTATE_ATLAS_PATH is required." >&2
    exit 1
fi
if [[ ! -r "$CELLSTATE_ATLAS_PATH" ]]; then
    echo "CELLSTATE_ATLAS_PATH must identify a readable file." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

export CELLSTATE_OPENAI_TIMEOUT_SECONDS="${CELLSTATE_OPENAI_TIMEOUT_SECONDS:-90}"
export CELLSTATE_OPENAI_MAX_RETRIES="${CELLSTATE_OPENAI_MAX_RETRIES:-2}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_DIR"
exec "$PYTHON_BIN" prototype_agent.py
