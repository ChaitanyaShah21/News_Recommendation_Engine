#!/usr/bin/env bash
# Run the released worker image with upstream `develop`'s compute_worker.py mounted
# over its own.
#
# Why this rather than patching line by line: every released tag is missing several
# fixes, and we found them one at a time, each costing a submission to discover.
# Mounting the whole upstream file picks up all of them at once. The image supplies
# the dependencies; the file supplies the logic.
#
# Safe to try: if the upstream file will not import against the image's installed
# packages, the container dies at startup, which costs nothing. Only a job actually
# taken from the queue costs a submission.
set -euo pipefail

SRC=https://raw.githubusercontent.com/codalab/codabench/develop/compute_worker/compute_worker.py

echo "==> Stopping any existing worker"
docker rm -f compute_worker 2>/dev/null || true

echo "==> Reclaiming stuck temp dirs"
sudo rm -rf /codabench/tmp* 2>/dev/null || true
df -h / | tail -1

echo "==> Fetching upstream compute_worker.py"
curl -fsSL "$SRC" -o /codabench/compute_worker.py
head -1 /codabench/compute_worker.py | grep -q "^import asyncio" \
    || { echo "FATAL: fetched file does not look like compute_worker.py"; exit 1; }
grep -q 'secret=str(run_args' /codabench/compute_worker.py \
    || { echo "FATAL: upstream no longer contains the secret fix"; exit 1; }
grep -q 'makedirs(self.output_dir' /codabench/compute_worker.py \
    || { echo "FATAL: upstream no longer contains the output_dir fix"; exit 1; }
echo "    both known fixes present"

echo "==> Starting worker with upstream file mounted"
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /codabench/compute_worker.py:/compute_worker.py:ro \
    -d --env-file /codabench/.env \
    --name compute_worker \
    --log-opt max-size=50m --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:latest

echo "==> Waiting for startup (import errors surface here, and cost nothing)"
sleep 10
if docker logs compute_worker 2>&1 | grep -qE "ImportError|ModuleNotFoundError|AttributeError|Traceback"; then
    echo "FAILED to start -- upstream file will not import in this image."
    echo "Fall back to: docker build -f codabench/Dockerfile.worker-patched -t worker-patched ."
    docker logs compute_worker 2>&1 | tail -20
    exit 1
fi
docker logs compute_worker 2>&1 | tail -8
echo
echo "If you see 'ready.' above, resubmit on Codabench and watch:"
echo "    docker logs -f compute_worker 2>&1 | tee ~/worker.log"
