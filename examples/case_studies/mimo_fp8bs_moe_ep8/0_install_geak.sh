#!/usr/bin/env bash
# Clone GEAK and install the Claude Code CLI (>= 2.1.177). Nothing is pip-installed into the SGLang env.
set -euo pipefail
GEAK_HOME="${GEAK_HOME:-/sgl-workspace/GEAK}"
MIN_CLAUDE=2.1.177
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -f "$GEAK_HOME/kernel_workflow/kernel_workflow.js" ]]; then
  git clone --depth 1 ${GEAK_REF:+--branch "$GEAK_REF"} https://github.com/AMD-AGI/GEAK.git "$GEAK_HOME"
fi
echo "GEAK: $GEAK_HOME ($(git -C "$GEAK_HOME" log -1 --format='%h %cd' --date=short))"

claude_ver() { claude --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1; }
cur="$(claude_ver || true)"
if [[ -z "$cur" || "$(printf '%s\n%s\n' "$MIN_CLAUDE" "$cur" | sort -V | head -1)" != "$MIN_CLAUDE" ]]; then
  # Same native installer GEAK's own bootstrap uses; installs to ~/.local/bin.
  curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | bash -s latest
  cur="$(claude_ver || true)"
fi
echo "Claude Code: ${cur:-NOT FOUND} (need >= $MIN_CLAUDE)"

for tool in git python3 flock rocprofv3 rocprof-compute; do
  printf '%-16s %s\n' "$tool" "$(command -v "$tool" || echo MISSING)"
done
