#!/usr/bin/env bash
# MANUAL Docker smoke test (requires a running Docker daemon).
set -euo pipefail

echo "== build =="
docker build -t finance-controller .

echo "== run =="
CID=$(docker run -d --name finance-controller-test -p 8000:8000 finance-controller)
cleanup() { docker rm -f finance-controller-test >/dev/null 2>&1 || true; }
trap cleanup EXIT

sleep 3   # let uvicorn start

echo "== health (no API key) =="
curl -fsS http://127.0.0.1:8000/health | grep '"status": *"ok"'

echo "== deterministic pipeline =="
curl -fsS -X POST http://127.0.0.1:8000/pipeline/run \
     -H 'Content-Type: application/json' -d '{}' \
| python -c "import json,sys; b=json.load(sys.stdin); assert b['case_count']==122; print('case_count:', b['case_count'])"

echo "== non-root check =="
UID_OUT=$(docker exec finance-controller-test id -u)
echo "container uid: $UID_OUT"
echo "UIDOUT"∣grep−q′10001UID_OUT" | grep -q '^10001UIDO​UT"∣grep−q′10001'

echo "== persistent volume mount =="
docker rm -f finance-controller-test >/dev/null
docker run -d --name finance-controller-test \
  -p 8000:8000 -v fc_runs_test:/app/data/runs finance-controller
sleep 3

curl -fsS -X POST http://127.0.0.1:8000/pipeline/save \
     -H 'Content-Type: application/json' -d '{"run_name":"smoke"}' \
| grep '"run_name": *"smoke"'

docker exec finance-controller-test ls /app/data/runs | grep -q 'smoke.json'
echo "volume persistence OK"

cleanup
trap - EXIT

echo "DOCKER SMOKE TEST PASSED"
