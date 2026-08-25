#!/usr/bin/env bash
# PostToolUse hook: auto-push the inspector site whenever Claude regenerates it.
#
# Fires after every Bash/Write/Edit, but only acts when the tool actually
# touched the inspector:
#   - a Bash command that runs scripts/build_data_inspector.py, or
#   - a Write/Edit to any file under data/plots/inspector/
# On a match it runs refresh_inspector_site.sh (which no-ops if nothing changed)
# and logs the result. Intended to be wired as an async hook so it never
# blocks the session while git pushes ~33 MB.
#
# Reads the hook payload as JSON on stdin. Always exits 0 (never blocks a tool).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="$PROJECT_ROOT/.inspector-refresh.log"

payload="$(cat)"
tool="$(printf '%s' "$payload" | jq -r '.tool_name // empty')"

matched=0
case "$tool" in
  Bash)
    cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty')"
    [[ "$cmd" == *build_data_inspector* ]] && matched=1
    ;;
  Write|Edit|StrReplace)
    fp="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty')"
    [[ "$fp" == *data/plots/inspector/* ]] && matched=1
    ;;
esac

[[ "$matched" -eq 1 ]] || exit 0

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  trigger=$tool ====="
  "$SCRIPT_DIR/refresh_inspector_site.sh"
} >>"$LOG" 2>&1

exit 0
