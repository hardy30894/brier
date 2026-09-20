#!/bin/bash
# sweep.sh <pod names...> : poll each pod; when it writes a DONE marker, pull (with adapters if it is a release pod) and terminate it
cd "$(dirname "$0")"
left=("$@")
for i in $(seq 1 240); do
  still=()
  for n in "${left[@]}"; do
    k=$(python3 pod.py --name "$n" exec "ls results/v2/*/DONE_* 2>/dev/null | wc -l" 2>/dev/null | tail -1)
    if [ "${k:-0}" -ge 1 ]; then
      case "$n" in rel-*) python3 pod.py --name "$n" pull --adapters 2>&1 | tail -1;; *) python3 pod.py --name "$n" pull 2>&1 | tail -1;; esac
      python3 pod.py --name "$n" down 2>&1 | grep -c terminated
      echo "SWEPT $n"
    else
      still+=("$n")
    fi
  done
  left=("${still[@]}")
  [ ${#left[@]} -eq 0 ] && break
  sleep 60
done
echo "SWEEP_DONE remaining: ${left[*]:-none}"
