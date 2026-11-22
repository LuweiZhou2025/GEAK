# Usage: source /sgl-workspace/geak_mimo_moe/1_env.sh
# Points Claude Code at the AMD LLM gateway. The key is read without echo and never written to disk.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file instead: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

if [[ -z "${ANTHROPIC_CUSTOM_HEADERS:-}" ]]; then
  read -rsp "AMD LLM gateway subscription key (input hidden): " _amd_key
  echo
  if [[ -z "$_amd_key" ]]; then
    echo "empty key" >&2
    unset _amd_key
    return 1
  fi
  # The gateway authenticates only via this header; a Bearer token gets 401 (GEAK interface/runtime/SETUP.md).
  export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${_amd_key}"
  unset _amd_key
fi

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://llm-api.amd.com/Unified}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-5-5}"
# Sub-agent / background model tiers must also be ids the gateway serves.
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-$ANTHROPIC_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-$ANTHROPIC_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-$ANTHROPIC_MODEL}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN GEAK_AMDKEY
# Claude Code will not start without a credential; the gateway itself only checks the subscription header.
if [[ "${GEAK_CLAUDE_AUTH:-bearer}" == x-api-key ]]; then
  export ANTHROPIC_API_KEY=amd-gateway
else
  export ANTHROPIC_AUTH_TOKEN=amd-gateway
fi
echo "Claude Code -> $ANTHROPIC_BASE_URL (model $ANTHROPIC_MODEL, placeholder auth: ${GEAK_CLAUDE_AUTH:-bearer})"
