#!/usr/bin/env bash
# Auto-approve the plugin's own engine calls, so one annotation run does not
# generate one permission prompt per measurement.
#
# A run issues dozens of distinct `lambdakit ...` command strings. Permission
# rules match the command, not the binary, so approving them one at a time
# means approving each subcommand separately — the run stops constantly.
#
# This grants exactly one thing: non-destructive `lambdakit` invocations.
# Everything else falls through to the normal prompt. Three guards keep that
# narrow:
#
#   1. the command must BE a lambdakit invocation, not merely contain one
#   2. no shell metacharacters, so nothing can be chained onto the back of an
#      approved command (`lambdakit status; rm -rf ~`)
#   3. subcommands that delete data still prompt, every time
#
# Silence (exit 0, no output) means "no opinion" — the normal flow decides.
set -uo pipefail

payload=$(cat)

decision=$("${LAMBDAKIT_PYTHON:-python3}" - "$payload" <<'PY'
import json, re, sys

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

if data.get("tool_name") != "Bash":
    sys.exit(0)

cmd = (data.get("tool_input") or {}).get("command", "").strip()
if not cmd:
    sys.exit(0)

# 2. Refuse to reason about anything with shell control characters. An approved
#    prefix must not be able to carry an unapproved payload behind it.
if re.search(r"[;&|`$><\n]|\$\(", cmd):
    sys.exit(0)

# 1. The command must start with the engine itself — bare `lambdakit`, or an
#    absolute path ending in /bin/lambdakit (the bundled entry point).
head = cmd.split()[0].strip('"\'')
if not (head == "lambdakit" or head.endswith("/bin/lambdakit")):
    sys.exit(0)

# 3. Anything that removes data keeps its prompt. The annotation pipeline never
#    calls these; a human asking for cleanup should still be asked.
rest = cmd.split()[1:]
DESTRUCTIVE = {("clean",), ("cache", "rm")}
words = tuple(w for w in rest if not w.startswith("-"))
if any(words[:len(d)] == d for d in DESTRUCTIVE):
    sys.exit(0)

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": (
            "lambda-annotate: read-only/analysis engine call, auto-approved by the "
            "plugin so an annotation run completes without a prompt per measurement. "
            "Commands that delete data are excluded and still prompt."
        ),
    }
}))
PY
) || exit 0

[ -n "$decision" ] && printf '%s\n' "$decision"
exit 0
