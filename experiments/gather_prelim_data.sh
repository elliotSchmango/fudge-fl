#!/usr/bin/env bash
set -euo pipefail

echo "clean port 8080..." #clean port
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
SEED="${SEED:-67}"
MALICIOUS_CLIENT_ID="${MALICIOUS_CLIENT_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-1}"
SERVER_ADDRESS="${SERVER_ADDRESS:-0.0.0.0:8080}"
CLIENT_SERVER_ADDRESS="${CLIENT_SERVER_ADDRESS:-127.0.0.1:8080}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: missing executable $PYTHON_BIN" >&2
    echo "Create/install dependencies in the venv, then rerun." >&2
    exit 1
fi

SERVER_PID=""
CLIENT_PIDS=()

cleanup() {
    if [[ ${#CLIENT_PIDS[@]} -gt 0 ]]; then
        kill "${CLIENT_PIDS[@]}" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting FUDGE-FL Server..."
"$PYTHON_BIN" "$ROOT_DIR/src/server.py" \
    --num-clients "$NUM_CLIENTS" \
    --malicious-client-id "$MALICIOUS_CLIENT_ID" \
    --seed "$SEED" \
    --unlearn-batch-size "$BATCH_SIZE" \
    --unlearn-epochs "$EPOCHS" \
    --server-address "$SERVER_ADDRESS" \
    --num-rounds 50 &
SERVER_PID=$!

#pause a few seconds so the server can initialize.
sleep 3
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server failed to start. See traceback above." >&2
    wait "$SERVER_PID"
fi

echo "Starting ${NUM_CLIENTS} FL Clients..."
for ((i=0; i<NUM_CLIENTS; i++)); do
    echo "Launching client $i..."
    "$PYTHON_BIN" "$ROOT_DIR/src/client.py" \
        --client-id "$i" \
        --malicious-client-id "$MALICIOUS_CLIENT_ID" \
        --num-clients "$NUM_CLIENTS" \
        --seed "$SEED" \
        --server-address "$CLIENT_SERVER_ADDRESS" &
    CLIENT_PIDS+=("$!")
done

echo "FUDGE-FL simulation is running. Waiting for 50 rounds to complete..."

#non-zero exits from server/clients will terminate this script (set -e).
wait "$SERVER_PID"
for pid in "${CLIENT_PIDS[@]}"; do
    wait "$pid"
done

echo "Simulation complete! Check the terminal for your Privacy, Utility, and Security scores."