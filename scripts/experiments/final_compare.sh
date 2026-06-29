#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
for c in nomanager replan0 replan1 replan5; do
  until [ -f runs/odysseys45_qwen37plus_$c/odysseys_eval_fulltext.json ]; do sleep 30; done
done
.venv/bin/python - <<'PY'
import json
def load(p):
    d=json.load(open(p)); t=d['tasks']
    return {x['task_id']:x['average_rubric_score'] for x in t}
print("=== odysseys45 qwen3.7-plus: capped@100  vs  FULL-text (max-steps 0, img 200) ===\n")
print(f"{'config':<11}{'capped rubric':>14}{'full rubric':>13}{'  ':>2}{'capped perf':>13}{'full perf':>11}")
for c in ['nomanager','replan0','replan1','replan5']:
    cap=load(f'runs/odysseys45_qwen37plus_{c}/odysseys_eval.json')
    full=load(f'runs/odysseys45_qwen37plus_{c}/odysseys_eval_fulltext.json')
    n=len(full)
    rc=sum(cap.values())/len(cap); rf=sum(full.values())/n
    pc=sum(1 for v in cap.values() if v>=1); pf=sum(1 for v in full.values() if v>=1)
    print(f"{c:<11}{rc:>14.3f}{rf:>13.3f}{'':>2}{str(pc)+'/45':>13}{str(pf)+'/45':>11}")
PY
echo "FINAL COMPARE DONE"
