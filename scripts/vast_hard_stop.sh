#!/usr/bin/env bash
set -euo pipefail
: "${VAST_INSTANCE_ID:?set VAST_INSTANCE_ID}"
: "${MAX_DOLLARS:?set MAX_DOLLARS}"
: "${VAST_DOLLARS_PER_HOUR:?set VAST_DOLLARS_PER_HOUR}"
started=${VAST_STARTED_EPOCH:-$(date +%s)}
seconds=$(awk -v budget="$MAX_DOLLARS" -v rate="$VAST_DOLLARS_PER_HOUR" \
    'BEGIN { printf "%d", 3600 * budget / rate }')
deadline=$((started + seconds))
remaining=$((deadline - $(date +%s)))
if (( remaining > 0 )); then sleep "$remaining"; fi
vastai destroy instance "$VAST_INSTANCE_ID"

