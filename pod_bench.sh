# usage: bash pod_bench.sh qwen|phi   -> correctness check, then server throughput at several batch sizes
cd /workspace/brier
if [ "$1" = phi ]; then M=microsoft/Phi-4-mini-instruct; A=adapters/ptr4/Phi-4-mini-instruct-brier-r16-s1000-u0.3-train-pointer; else M=Qwen/Qwen3-0.6B; A=adapters/ptr/Qwen3-0.6B-brier-r16-s5000-u0.3-train-pointer; fi
python3 check_batch.py --model $M --adapter $A 2>&1 | grep -v Warn | tail -3
python3 check_batch.py --model $M --adapter $A --capture 2>&1 | grep -v Warn | tail -3
python3 speed_pointer.py --model $M --adapter $A --capture 2>&1 | grep -v Warn | tail -4
run() {  # batch label
  pkill -f '^python3 -m brier.serve' ; sleep 2
  nohup python3 -m brier.serve --model $M --adapter $A --capture --batch $1 --port 8000 > serve_$2.log 2>&1 &
  for i in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
  grep -h "captured\|serving" serve_$2.log | tail -2
  for c in 1 8 32; do python3 bench_serve.py --clients $c --n 30 --questions 3 | sed "s/^/$2 /"; done
  python3 bench_serve.py --clients 8 --n 30 --questions 10 | sed "s/^/$2 /"
}
run 1 $1-b1
run 8 $1-b8
run 16 $1-b16
pkill -f '^python3 -m brier.serve'
echo BENCH_DONE
