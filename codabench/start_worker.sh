#!/usr/bin/env bash
# Start ONE Codabench compute worker for the EB-NeRD / RecSys 2024 competition (2469).
#
# Follows https://github.com/jppol-ai/ebnerd-benchmark/tree/main/codabench exactly,
# with one deliberate difference: the organizers' script starts three workers on a
# 16 GB t3.xlarge. We start one, because each concurrent scoring job is a separate
# container reading 13.5M prediction rows and we have 12 GB.
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] || { echo "ERROR: codabench/.env missing"; exit 1; }
[ -d /codabench ] || { echo "ERROR: /codabench missing -- run setup_host_dir.sh first"; exit 1; }

docker rm -f compute_worker 2>/dev/null || true
docker pull codalab/competitions-v2-compute-worker:cpu1.1

docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:cpu1.1

echo "Worker started. Follow it with:  docker logs -f compute_worker"
