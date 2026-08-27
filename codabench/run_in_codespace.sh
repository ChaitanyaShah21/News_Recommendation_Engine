#!/usr/bin/env bash
# Start ONE Codabench compute worker inside a GitHub Codespace.
#
# Differs from bootstrap_vm.sh in two ways that matter:
#   - Codespaces has no systemd, so there is no `systemctl enable docker`.
#     Docker is already running via the docker-in-docker feature; we verify
#     rather than install.
#   - sudo is passwordless here, so /codabench needs no interactive prompt.
set -euo pipefail

echo "==> Pre-flight"
ARCH=$(uname -m)
echo "    architecture : $ARCH"
[ "$ARCH" = "x86_64" ] || { echo "FATAL: worker image is amd64-only, this host is $ARCH"; exit 1; }

python3 - <<'PY' || { echo "FATAL: cannot reach the broker port"; exit 1; }
import socket, sys
s = socket.socket(); s.settimeout(10)
try:
    s.connect(("www.codabench.org", 5672)); print("    port 5672    : OPEN")
except Exception as e:
    print(f"    port 5672    : FAILED -> {type(e).__name__}"); sys.exit(1)
finally:
    s.close()
PY

MEM_GB=$(awk '/^MemTotal:/ {printf "%.0f", $2/1048576}' /proc/meminfo)
echo "    memory       : ${MEM_GB}G  (organizers used 16G)"
if [ "$MEM_GB" -lt 12 ]; then
    echo
    echo "    WARNING: this looks like a 2-core codespace. Scoring reads 13.5M"
    echo "    prediction rows; the organizers ran it on 16G. Recreate with the"
    echo "    4-core / 16GB machine type before starting a real run."
    read -rp "    Continue anyway? [y/N] " ok
    [ "$ok" = "y" ] || exit 1
fi

DISK_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "    free disk    : ${DISK_GB}G  (worker downloads the submission + hidden test set)"

docker info --format '    docker       : {{.ServerVersion}}' 2>/dev/null \
    || { echo "FATAL: docker not reachable in this codespace"; exit 1; }

echo "==> Creating /codabench"
sudo mkdir -p /codabench && sudo chown "$USER:$USER" /codabench

echo "==> Fetching the organizers' published queue credentials"
curl -fsSL https://raw.githubusercontent.com/jppol-ai/ebnerd-benchmark/main/codabench/.env -o /codabench/.env
grep -q '^BROKER_URL=' /codabench/.env || { echo "FATAL: .env fetch returned junk"; exit 1; }
echo "    got $(wc -l < /codabench/.env) lines"

echo "==> Starting ONE worker"
docker rm -f compute_worker 2>/dev/null || true
docker pull codalab/competitions-v2-compute-worker:cpu1.1
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file /codabench/.env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:cpu1.1

echo
echo "Worker started. Now run this and LEAVE IT RUNNING:"
echo
echo "    docker logs -f compute_worker"
echo
echo "That stream is doing two jobs: showing you whether it picked up YOUR"
echo "submission or another participant's, and generating the terminal output"
echo "that resets the codespace idle timer."
