#!/usr/bin/env bash
# 30-second probe: can THIS host run the worker at all? Safe to run anywhere
# (Codespace, VM, laptop). Changes nothing.
echo "architecture : $(uname -m)   (must be x86_64)"
echo "memory       : $(free -h | awk '/^Mem:/ {print $2}')   (organizers used 16G)"
echo -n "docker       : "; docker info --format '{{.ServerVersion}}' 2>/dev/null || echo "not reachable"
python3 - <<'PY'
import socket
for port in (5672, 443):
    s = socket.socket(); s.settimeout(8)
    try:
        s.connect(("www.codabench.org", port)); print(f"port {port}     : OPEN")
    except Exception as e:
        print(f"port {port}     : FAILED -> {type(e).__name__}")
    finally:
        s.close()
PY
