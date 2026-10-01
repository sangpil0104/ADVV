#!/bin/sh
# Print every GPU with total/used memory and the OS user of each compute process.
# Read-only: never kills or launches anything. Occupancy changes often; rerun right before launching.
set -eu
nvidia-smi --query-gpu=index,gpu_uuid,memory.total,memory.used --format=csv,noheader,nounits |
while IFS=, read -r idx uuid total used; do
  idx=$(echo "$idx" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' ')
  owners=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
    while IFS=, read -r u pid mem; do
      u=$(echo "$u" | tr -d ' '); pid=$(echo "$pid" | tr -d ' ')
      [ "$u" = "$uuid" ] || continue
      user=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
      printf '%s(pid %s, %sMiB) ' "$user" "$pid" "$(echo "$mem" | tr -d ' ')"
    done)
  printf 'GPU%s total=%sMiB used=%sMiB free=%sMiB owners: %s\n' \
    "$idx" "$(echo "$total" | tr -d ' ')" "$(echo "$used" | tr -d ' ')" \
    "$(( $(echo "$total" | tr -d ' ') - $(echo "$used" | tr -d ' ') ))" "${owners:-none}"
done
echo "me=$(id -un)"
