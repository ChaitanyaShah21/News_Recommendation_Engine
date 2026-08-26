#!/usr/bin/env bash
# One-shot setup for a FRESH x86-64 Ubuntu VM (Azure Standard_D4s_v3 or similar).
# Run as a normal user with sudo. Installs Docker, fetches the organizers' queue
# credentials, and starts exactly ONE Codabench compute worker.
#
#   curl -fsSL <this file> | bash      -- or just paste it after ssh'ing in.
set -euo pipefail

echo "==> Pre-flight (these are the two things that killed the local attempt)"
ARCH=$(uname -m)
echo "    architecture: $ARCH"
[ "$ARCH" = "x86_64" ] || { echo "FATAL: worker image is amd64-only, this host is $ARCH"; exit 1; }

python3 - <<'PY' || { echo "FATAL: cannot reach the broker port -- network filters AMQP"; exit 1; }
import socket, sys
s = socket.socket(); s.settimeout(10)
try:
    s.connect(("www.codabench.org", 5672)); print("    port 5672: OPEN")
except Exception as e:
    print(f"    port 5672: FAILED -> {type(e).__name__}: {e}"); sys.exit(1)
finally:
    s.close()
PY

echo "==> Memory (organizers used 16 GB; scoring reads 13.5M prediction rows)"
free -h | awk '/^Mem:/ {print "    total: " $2}'

echo "==> Installing Docker"
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
fi
sudo systemctl enable --now docker

echo "==> Creating /codabench"
sudo mkdir -p /codabench && sudo chown "$USER:$USER" /codabench

echo "==> Fetching the organizers' published queue credentials"
# Straight from github.com/jppol-ai/ebnerd-benchmark -- not ours, and not stored in git.
curl -fsSL https://raw.githubusercontent.com/jppol-ai/ebnerd-benchmark/main/codabench/.env -o /codabench/.env
grep -q '^BROKER_URL=' /codabench/.env || { echo "FATAL: .env fetch returned junk"; exit 1; }
echo "    got $(wc -l < /codabench/.env) lines"

echo "==> Starting ONE worker (organizers start three; they had 16 GB per three jobs)"
sudo docker rm -f compute_worker 2>/dev/null || true
sudo docker pull codalab/competitions-v2-compute-worker:cpu1.1
sudo docker run \
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
echo "Done. Watch it pick up a job with:"
echo "    sudo docker logs -f compute_worker"
echo "Remember: the queue is SHARED -- the first job it takes may be someone else's."
