#!/usr/bin/env bash
# Quick vLLM multimodal diagnostic — run on GPU node
set -euo pipefail

EVAL_PORT="${1:-8000}"
MODEL="${2:-Vision-OPD-4B}"
IMAGE="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/third_party/Vision-OPD/eval/VStar_images/1.jpg"

echo "=== 1. Health check ==="
curl -s "http://localhost:${EVAL_PORT}/health" && echo " OK" || echo " FAIL"

echo ""
echo "=== 2. Text-only request ==="
TEXT_RESP=$(curl -s -w "\n%{http_code}" "http://localhost:${EVAL_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":10}")
HTTP_CODE=$(echo "$TEXT_RESP" | tail -1)
BODY=$(echo "$TEXT_RESP" | sed '$d')
echo "HTTP ${HTTP_CODE}"
echo "${BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))" 2>/dev/null || echo "${BODY}" | head -5

echo ""
echo "=== 3. Multimodal request (data URI) ==="
B64=$(python3 -c "
import base64
with open('${IMAGE}','rb') as f:
    print(base64.b64encode(f.read()).decode())
")
MM_RESP=$(curl -s -w "\n%{http_code}" "http://localhost:${EVAL_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${B64}\"}},{\"type\":\"text\",\"text\":\"Describe this image briefly.\"}]}],\"max_tokens\":20}")
HTTP_CODE=$(echo "$MM_RESP" | tail -1)
BODY=$(echo "$MM_RESP" | sed '$d')
echo "HTTP ${HTTP_CODE}"
echo "${BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))" 2>/dev/null || echo "${BODY}" | head -20

echo ""
echo "=== 4. Multimodal request (local file path) ==="
FP_RESP=$(curl -s -w "\n%{http_code}" "http://localhost:${EVAL_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"file://${IMAGE}\"}},{\"type\":\"text\",\"text\":\"Describe this image briefly.\"}]}],\"max_tokens\":20}")
HTTP_CODE=$(echo "$FP_RESP" | tail -1)
BODY=$(echo "$FP_RESP" | sed '$d')
echo "HTTP ${HTTP_CODE}"
echo "${BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))" 2>/dev/null || echo "${BODY}" | head -20

echo ""
echo "=== Done ==="
