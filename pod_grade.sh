# Grade questions ride the row scorer; measure batching across requests on that path.
cd /workspace/brier
M=Qwen/Qwen3-0.6B; A=adapters/ptr/Qwen3-0.6B-brier-r16-s5000-u0.3-train-pointer
python3 check_batch.py --model $M --adapter $A --fp32 2>&1 | grep -a 'diff\|OK\|Error'
python3 check_batch.py --model $M --adapter $A --capture 2>&1 | grep -a 'diff\|OK\|Error'
run() {
  pkill -f '^python3 -m brier.serve' ; sleep 2
  nohup python3 -m brier.serve --model $M --adapter $A --capture --batch $1 --port 8000 > serve_grade_$2.log 2>&1 &
  for i in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
  for c in 1 8 32; do python3 bench_serve.py --clients $c --n 30 --questions 3 --grade | sed "s/^/$2 /"; done
  python3 bench_serve.py --clients 8 --n 30 --questions 10 --grade | sed "s/^/$2 /"
}
run 1 b1
run 8 b8
pkill -f '^python3 -m brier.serve'
echo GRADE_DONE
