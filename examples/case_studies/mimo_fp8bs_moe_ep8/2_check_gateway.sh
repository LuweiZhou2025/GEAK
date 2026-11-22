#!/usr/bin/env bash
# Check that the AMD LLM gateway answers Anthropic /v1/messages and that Claude Code can use it.
# Run after: source 1_env.sh
set -euo pipefail
: "${ANTHROPIC_CUSTOM_HEADERS:?run 'source 1_env.sh' first}"
url="${ANTHROPIC_BASE_URL%/}/v1/messages"
body="{\"model\":\"$ANTHROPIC_MODEL\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK\"}]}"
# Same headers Claude Code will send, passed via a stdin config so the key never shows up in `ps`.
auth_headers() {
  printf 'header = "%s"\n' "$ANTHROPIC_CUSTOM_HEADERS"
  if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then printf 'header = "Authorization: Bearer %s"\n' "$ANTHROPIC_AUTH_TOKEN"; fi
  if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then printf 'header = "x-api-key: %s"\n' "$ANTHROPIC_API_KEY"; fi
}
resp="$(mktemp)"
code="$(auth_headers | curl -sS -K - -o "$resp" -w '%{http_code}' \
  -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' -d "$body" "$url")" || true
echo "HTTP $code  $url  (model $ANTHROPIC_MODEL)"
head -c 300 "$resp"
echo
rm -f "$resp"
if [[ "$code" != 200 ]]; then
  echo "gateway check failed; if only the placeholder auth header breaks it, retry: GEAK_CLAUDE_AUTH=x-api-key source 1_env.sh" >&2
  exit 1
fi

claude --version
timeout 300 claude -p "Reply with exactly: OK"
