#!/usr/bin/env bash
# One-time: create the worker's cache directory. Needs sudo because it lives at /.
# The path must match HOST_DIRECTORY in .env and the -v mount in start_worker.sh:
# the worker asks the Docker daemon to mount these paths into sibling containers,
# and the daemon resolves them against the host, not against the worker.
set -euo pipefail
sudo mkdir -p /codabench
sudo chown "$USER:$USER" /codabench
echo "/codabench ready:"; ls -ld /codabench
