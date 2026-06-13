#!/usr/bin/env python3
"""Analyze and visualize a single MACU task run directory.

Input: one task folder from a run, e.g.
    runs/subset36_replan5/22a4636f-8179-4357-8e87-d1743ece1f81

Produces:
  1. A terminal report of the algorithm flow as a timeline: initial graph ->
     each replan (with the exact add/remove/modify/cancel graph ops) -> final
     aggregation -> evaluator score. The focus is making the DAG evolution
     legible.
  2. A self-contained HTML file (<task_dir>/analysis.html) that renders every
     graph snapshot as a DAG (nodes + dependency edges, colored by status) so
     you can step through how the graph changed across replans. Zero external
     deps: layout + SVG are done in plain JS, all data inlined.

Usage:
    python scripts/analyze_task.py runs/<run>/<task_id>
    python scripts/analyze_task.py runs/<run>/<task_id> --no-html
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _graph_body(snapshot: dict) -> dict:
    """A snapshot may wrap the graph under 'dependency_graph' or be the graph."""
    if "dependency_graph" in snapshot:
        return snapshot["dependency_graph"]
    return snapshot


def _nodes_of(graph_body: dict) -> dict[str, dict]:
    """Return {id: node} for all subtasks + aggregation node."""
    nodes: dict[str, dict] = {}
    for st in graph_body.get("subtasks", []) or []:
        nodes[st["id"]] = st
    agg = graph_body.get("aggregation")
    if agg and agg.get("id"):
        nodes[agg["id"]] = agg
    return nodes


def load_snapshots(task_dir: Path) -> list[dict]:
    """Load graph snapshots in chronological (numeric) order."""
    snap_dir = task_dir / "graph_snapshots"
    out: list[dict] = []
    if not snap_dir.is_dir():
        # fall back to the single dependency_graph.json
        g = _load_json(task_dir / "dependency_graph.json")
        if g is not None:
            out.append({"file": "dependency_graph.json", "seq": 0, "label": "final", "data": g})
        return out
    files = []
    for p in snap_dir.glob("dependency_graph_*.json"):
        m = re.match(r"dependency_graph_(\d+)_(.+)\.json", p.name)
        if m:
            files.append((int(m.group(1)), m.group(2), p))
    files.sort(key=lambda x: x[0])
    for seq, label, p in files:
        data = _load_json(p)
        if data is not None:
            out.append({"file": p.name, "seq": seq, "label": label, "data": data})
    return out


# --------------------------------------------------------------------------- #
# Status inference
# --------------------------------------------------------------------------- #
def node_status_map(final_results: Optional[dict]) -> dict[str, str]:
    """id -> 'success' | 'error' | 'unknown' from final_results.subtask_results."""
    out: dict[str, str] = {}
    if not final_results:
        return out
    for sid, info in (final_results.get("subtask_results") or {}).items():
        if isinstance(info, dict):
            out[sid] = info.get("status", "unknown")
    return out


def node_response_map(final_results: Optional[dict]) -> dict[str, str]:
    """id -> the subtask's self-reported response/error (the proximate cause
    that the manager reacted to when it decided to spawn a retry)."""
    out: dict[str, str] = {}
    if not final_results:
        return out
    for sid, info in (final_results.get("subtask_results") or {}).items():
        if isinstance(info, dict):
            out[sid] = info.get("response", "") or ""
    return out


# --------------------------------------------------------------------------- #
# Terminal report
# --------------------------------------------------------------------------- #
C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
    "cyan": "\033[36m", "blue": "\033[34m", "mag": "\033[35m",
}


def _c(s: str, color: str, use_color: bool) -> str:
    if not use_color:
        return s
    return f"{C[color]}{s}{C['reset']}"


def _trunc(s: str, n: int = 90) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _wrap(s: str, width: int = 88, indent: str = "        ") -> str:
    """Word-wrap text to width with a hanging indent (for full reasoning)."""
    import textwrap
    s = " ".join(str(s).split())
    lines = textwrap.wrap(s, width=width) or [""]
    return ("\n" + indent).join(lines)


def print_report(task_dir: Path, use_color: bool = True) -> None:
    fr = _load_json(task_dir / "final_results.json")
    summary = _load_json(task_dir / "summary.json")
    snapshots = load_snapshots(task_dir)
    replans = _load_jsonl(task_dir / "replan_log.jsonl")
    status = node_status_map(fr)
    responses = node_response_map(fr)

    def h(s):  # header
        return _c(s, "bold", use_color)

    print()
    print(h("═" * 78))
    print(h(f" MACU Task Analysis: {task_dir.name}"))
    print(h("═" * 78))

    # --- overview ---
    if fr:
        score = fr.get("osworld_evaluator_score")
        sc_str = ("n/a" if score is None else f"{score}")
        sc_col = "green" if isinstance(score, (int, float)) and score >= 1.0 else (
            "yellow" if isinstance(score, (int, float)) and score > 0 else "red")
        print(f"  Domain      : {fr.get('domain','?')}")
        print(f"  Instruction : {_trunc(fr.get('instruction',''), 100)}")
        print(f"  Eval score  : {_c(sc_str, sc_col, use_color)}")
        rp = fr.get("replanning", {}) or {}
        print(f"  Replanning  : {rp.get('num_applied','?')} applied / "
              f"{rp.get('num_calls','?')} calls / {rp.get('snapshot_count','?')} snapshots")
        dur = fr.get("total_duration_seconds")
        inf = fr.get("total_inference_seconds")
        if dur is not None:
            print(f"  Wall-clock  : {dur:.0f}s total ({inf:.0f}s in model inference)")
    if summary:
        print(f"  Cost        : ${summary.get('total_cost_usd',0):.4f} "
              f"(CUA ${summary.get('total_cua_cost_usd',0):.4f} / "
              f"Manager ${summary.get('total_manager_cost_usd',0):.4f})")

    # --- graph evolution timeline ---
    print()
    print(h("─" * 78))
    print(h(" GRAPH EVOLUTION  (how the manager's DAG changed over time)"))
    print(h("─" * 78))

    if snapshots:
        init = _graph_body(snapshots[0]["data"])
        nodes0 = _nodes_of(init)
        cua0 = [n for n in nodes0.values() if n.get("agent_type") == "cua"]
        print(f"\n{_c('① INITIAL GRAPH', 'cyan', use_color)}  "
              f"({snapshots[0]['file']})")
        analysis = init.get("task_analysis") or (
            snapshots[0]["data"].get("dependency_graph", {}).get("task_analysis"))
        if analysis:
            print(f"   manager's plan: {_trunc(analysis, 120)}")
        print(f"   {len(cua0)} CUA subtask(s) + 1 aggregation node:")
        for n in cua0:
            deps = n.get("dependencies") or []
            dep_s = f" ← depends on {deps}" if deps else " (no deps, can start immediately)"
            print(f"     • {_c(n['id'], 'blue', use_color)}: {n.get('description','')}{dep_s}")
            instr = n.get("instruction")
            if instr:
                print(f"       {_c('instruction:', 'dim', use_color)} {_wrap(instr, indent='         ')}")
        agg = init.get("aggregation", {})
        if agg:
            print(f"     ⊕ {_c(agg.get('id','final_aggregation'), 'mag', use_color)} "
                  f"(manager) ← depends on {agg.get('dependencies', [])}")

    # --- each replan ---
    if replans:
        print(f"\n{_c('② REPLANS', 'cyan', use_color)}  "
              f"(manager re-examines after each subtask finishes)")
        for r in replans:
            it = r.get("iteration")
            trig = r.get("trigger_sid")
            applied = r.get("applied")
            reason = r.get("reason")
            tag = (_c("APPLIED", "green", use_color) if applied
                   else _c("no change", "dim", use_color))
            print(f"\n   {_c(f'replan #{it}', 'yellow', use_color)} "
                  f"[trigger: {trig}, {reason}] → {tag}")
            # WHY (1): the proximate cause — what the triggering subtask reported
            # back, which is what the manager reacted to.
            if trig and trig in responses:
                tstat = status.get(trig, "?")
                tcol = "green" if tstat == "success" else ("red" if tstat == "error" else "dim")
                resp = responses.get(trig, "")
                print(f"      trigger outcome: [{_c(tstat, tcol, use_color)}] "
                      f"{_trunc(resp, 100)}")
            dec = r.get("decision")
            if not applied or not dec:
                if dec and dec.get("reasoning"):
                    print(f"      manager reasoning: {_wrap(dec['reasoning'])}")
                else:
                    print(f"      {_c('(manager decided the graph was fine as-is — no retry needed)', 'dim', use_color)}")
                continue
            # WHY (2): the manager's full deliberation — why a retry, and what's
            # different about it. Shown in full (not truncated).
            if dec.get("reasoning"):
                print(f"      manager reasoning: {_wrap(dec['reasoning'])}")
            # the four graph ops
            for op_name, col, sym in (("add", "green", "+"), ("remove", "red", "−"),
                                       ("cancel", "red", "✗")):
                items = dec.get(op_name) or []
                for it_obj in items:
                    if isinstance(it_obj, dict):
                        nid = it_obj.get("id", "?")
                        desc = it_obj.get("description", "")
                        extra = ""
                        if it_obj.get("variant_of"):
                            extra = f" (variant of {it_obj['variant_of']})"
                        label = "add (retry of " + str(trig) + ")" if op_name == "add" else op_name
                        print(f"      {_c(sym + ' ' + label, col, use_color)} "
                              f"{_c(nid, 'blue', use_color)}: {desc}{extra}")
                        instr = it_obj.get("instruction")
                        if op_name == "add" and instr:
                            print(f"        {_c('instruction:', 'dim', use_color)} {_wrap(instr, indent='          ')}")
                    else:
                        print(f"      {_c(sym + ' ' + op_name, col, use_color)} {it_obj}")
            mod = dec.get("modify") or {}
            for nid, changes in mod.items():
                ch = []
                if "dependencies" in changes:
                    ch.append(f"deps→{changes['dependencies']}")
                if "instruction" in changes:
                    ch.append("instruction rewritten")
                print(f"      {_c('~ modify', 'yellow', use_color)} "
                      f"{_c(nid, 'blue', use_color)}: {', '.join(ch)}")

    # --- final nodes + statuses ---
    print(f"\n{_c('③ FINAL STATE', 'cyan', use_color)}")
    if snapshots:
        final_body = _graph_body(snapshots[-1]["data"])
        for nid, n in _nodes_of(final_body).items():
            st = status.get(nid, "—")
            st_col = "green" if st == "success" else ("red" if st == "error" else "dim")
            kind = n.get("agent_type", "?")
            print(f"     [{_c(st.upper().ljust(7), st_col, use_color)}] "
                  f"{_c(nid, 'blue' if kind=='cua' else 'mag', use_color)} ({kind})")
    if fr and fr.get("final_response"):
        print(f"\n   {_c('Final response:', 'bold', use_color)}")
        print(f"   {_trunc(fr['final_response'], 300)}")

    print()
    print(h("═" * 78))


# --------------------------------------------------------------------------- #
# HTML visualization
# --------------------------------------------------------------------------- #
def build_html(task_dir: Path) -> str:
    fr = _load_json(task_dir / "final_results.json")
    snapshots = load_snapshots(task_dir)
    replans = _load_jsonl(task_dir / "replan_log.jsonl")
    status = node_status_map(fr)
    responses = node_response_map(fr)

    # Build a compact per-snapshot payload for the JS renderer.
    frames = []
    for snap in snapshots:
        body = _graph_body(snap["data"])
        nodes = []
        for st in body.get("subtasks", []) or []:
            nodes.append({
                "id": st["id"], "kind": "cua",
                "desc": st.get("description", ""),
                "instr": st.get("instruction", ""),
                "deps": st.get("dependencies", []) or [],
                "variant_of": st.get("variant_of"),
            })
        agg = body.get("aggregation")
        if agg and agg.get("id"):
            nodes.append({
                "id": agg["id"], "kind": "manager",
                "desc": agg.get("description", ""),
                "instr": agg.get("instruction", ""),
                "deps": agg.get("dependencies", []) or [],
            })
        frames.append({
            "seq": snap["seq"], "file": snap["file"], "label": snap["label"],
            "nodes": nodes,
        })

    payload = {
        "task_id": task_dir.name,
        "domain": (fr or {}).get("domain", ""),
        "instruction": (fr or {}).get("instruction", ""),
        "score": (fr or {}).get("osworld_evaluator_score"),
        "final_response": (fr or {}).get("final_response", ""),
        "status": status,
        "responses": {k: (v or "")[:240] for k, v in responses.items()},
        "frames": frames,
        "replans": replans,
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    # Plain JS DAG: layer nodes by longest-path depth, draw SVG edges.
    return _HTML_TEMPLATE.replace("__DATA__", data_json).replace(
        "__TITLE__", _html.escape(task_dir.name))


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>MACU Task — __TITLE__</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       margin:0;background:#0f1419;color:#e6e6e6;}
  header{padding:16px 24px;background:#1a212b;border-bottom:1px solid #2a3441;}
  h1{font-size:16px;margin:0 0 6px;}
  .meta{font-size:13px;color:#9aa7b5;line-height:1.5;}
  .score{font-weight:bold;padding:2px 8px;border-radius:4px;}
  .s-ok{background:#1f5132;color:#7ee2a8;} .s-fail{background:#5a2030;color:#ffa0b0;}
  .s-part{background:#5a4a1f;color:#ffd97a;}
  .wrap{display:flex;flex-direction:column;gap:0;}
  .controls{padding:12px 24px;background:#161d26;border-bottom:1px solid #2a3441;
            display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:5;}
  button{background:#2a3441;color:#e6e6e6;border:1px solid #3a4757;border-radius:6px;
         padding:6px 12px;cursor:pointer;font-size:13px;}
  button:hover{background:#34404f;} button:disabled{opacity:.4;cursor:default;}
  #frameLabel{font-size:14px;font-weight:bold;color:#7fd0ff;min-width:340px;}
  .stage{display:flex;gap:0;}
  #graph{flex:1;min-height:480px;padding:20px;overflow:auto;}
  #side{width:380px;background:#141a22;border-left:1px solid #2a3441;padding:16px 18px;
        font-size:13px;line-height:1.5;max-height:calc(100vh - 120px);overflow:auto;}
  .op{margin:4px 0;padding:6px 8px;border-radius:5px;font-size:12.5px;}
  .op-add{background:#15331f;border-left:3px solid #3fbf6f;}
  .op-mod{background:#33301a;border-left:3px solid #d4b14a;}
  .op-rm{background:#331a20;border-left:3px solid #d45a6a;}
  .op-none{background:#20262f;border-left:3px solid #555;color:#9aa7b5;}
  .reason{color:#9aa7b5;font-style:italic;margin:6px 0;font-size:12px;}
  svg text{fill:#e6e6e6;font-size:12px;}
  .legend{font-size:12px;color:#9aa7b5;display:flex;gap:16px;}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle;}
  h3{font-size:13px;margin:14px 0 6px;color:#7fd0ff;border-bottom:1px solid #2a3441;padding-bottom:4px;}
  .resp{white-space:pre-wrap;color:#cdd6df;font-size:12px;background:#10161e;padding:8px;border-radius:5px;}
  .instr{white-space:pre-wrap;color:#bcc7d2;font-size:11.5px;background:#10161e;padding:7px;border-radius:5px;margin-top:3px;line-height:1.45;}
</style></head>
<body>
<header>
  <h1>MACU Task Analysis — <span id="tid"></span></h1>
  <div class="meta" id="hdrMeta"></div>
</header>
<div class="controls">
  <button id="prev">◀ Prev</button>
  <span id="frameLabel"></span>
  <button id="next">Next ▶</button>
  <span class="legend">
    <span><span class="dot" style="background:#4a90d9"></span>CUA subtask</span>
    <span><span class="dot" style="background:#b86fd4"></span>Manager/aggregation</span>
    <span><span class="dot" style="background:#3fbf6f"></span>success</span>
    <span><span class="dot" style="background:#d45a6a"></span>error</span>
    <span><span class="dot" style="background:#c9a227"></span>pending (not run yet)</span>
    <span><span class="dot" style="background:#7a8694"></span>not-run/unknown</span>
  </span>
</div>
<div class="stage">
  <div id="graph"></div>
  <div id="side"></div>
</div>
<script>
const DATA = __DATA__;
let cur = 0;

document.getElementById('tid').textContent = DATA.task_id;
(function(){
  const s = DATA.score;
  let cls='s-part', txt = (s===null||s===undefined)?'n/a':String(s);
  if (typeof s==='number'){ if(s>=1) cls='s-ok'; else if(s<=0) cls='s-fail'; }
  document.getElementById('hdrMeta').innerHTML =
    `<b>domain:</b> ${esc(DATA.domain)} &nbsp;|&nbsp; <b>score:</b> `+
    `<span class="score ${cls}">${esc(txt)}</span><br><b>task:</b> ${esc(DATA.instruction)}`;
})();

function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

// longest-path layering for a clean left->right DAG
function layout(nodes){
  const byId={}; nodes.forEach(n=>byId[n.id]=n);
  const depth={};
  function d(id,seen){
    if(depth[id]!=null) return depth[id];
    if(seen.has(id)) return 0;
    seen.add(id);
    const n=byId[id]; if(!n||!n.deps.length){depth[id]=0;return 0;}
    let m=0; n.deps.forEach(p=>{ if(byId[p]) m=Math.max(m,d(p,seen)+1); });
    depth[id]=m; return m;
  }
  nodes.forEach(n=>d(n.id,new Set()));
  const layers={}; nodes.forEach(n=>{(layers[depth[n.id]]=layers[depth[n.id]]||[]).push(n);});
  return {depth,layers};
}

// Frame-aware status. The aggregation (manager) node only executes at the very
// end, so it must NOT show its final 'success' in intermediate frames — it is
// 'pending' until the last frame.
function effStatus(node, idx){
  const isLast = (idx === DATA.frames.length - 1);
  if(node.kind==='manager' && !isLast) return 'pending';
  return DATA.status[node.id] || 'unknown';
}
function colorOf(s){
  if(s==='success') return '#3fbf6f';
  if(s==='error') return '#d45a6a';
  if(s==='pending') return '#c9a227';   // amber = not run yet
  return '#7a8694';                       // unknown / not-run
}

function renderGraph(frame, idx){
  const nodes=frame.nodes;
  const {depth,layers}=layout(nodes);
  const colW=260, rowH=92, padX=30, padY=30, boxW=210, boxH=64;
  const maxLayer=Math.max(0,...Object.keys(layers).map(Number));
  let maxRows=1; Object.values(layers).forEach(a=>maxRows=Math.max(maxRows,a.length));
  const W=padX*2+(maxLayer+1)*colW, H=padY*2+maxRows*rowH;
  const pos={};
  Object.entries(layers).forEach(([L,arr])=>{
    arr.forEach((n,i)=>{ pos[n.id]={x:padX+(+L)*colW, y:padY+i*rowH}; });
  });
  let edges='';
  nodes.forEach(n=>n.deps.forEach(p=>{
    if(!pos[p]) return;
    const a=pos[p], b=pos[n.id];
    const x1=a.x+boxW, y1=a.y+boxH/2, x2=b.x, y2=b.y+boxH/2;
    const mx=(x1+x2)/2;
    edges+=`<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" `+
           `stroke="#4a5868" stroke-width="2" fill="none" marker-end="url(#arr)"/>`;
  }));
  let boxes='';
  nodes.forEach(n=>{
    const p=pos[n.id]; const isCua=n.kind==='cua';
    const fill=isCua?'#1d2b3d':'#2e1d3d';
    const stroke=isCua?'#4a90d9':'#b86fd4';
    const sc=colorOf(effStatus(n, idx));
    boxes+=`<g>
      <rect x="${p.x}" y="${p.y}" width="${boxW}" height="${boxH}" rx="8"
            fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>
      <circle cx="${p.x+12}" cy="${p.y+14}" r="5" fill="${sc}"/>
      <text x="${p.x+24}" y="${p.y+18}" font-weight="bold">${esc(n.id)}</text>
      <text x="${p.x+10}" y="${p.y+38}" font-size="11" fill="#aebaccc">${esc(trunc(n.desc,30))}</text>
      ${n.variant_of?`<text x="${p.x+10}" y="${p.y+54}" font-size="10" fill="#d4b14a">variant of ${esc(n.variant_of)}</text>`:''}
    </g>`;
  });
  return `<svg width="${W}" height="${H}">
    <defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3"
      orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4a5868"/></marker></defs>
    ${edges}${boxes}</svg>`;
}
function trunc(s,n){s=(s||'').replace(/\s+/g,' ');return s.length<=n?s:s.slice(0,n-1)+'…';}

// map a frame index to the replan decision that produced it (by seq)
function replanForFrame(frame){
  // snapshot label like "after_subtask_1_retry"; match replan whose applied decision
  // produced this. We align by order: frame[0]=initial, frame[k] follows the k-th APPLIED replan.
  return null;
}

function renderSide(idx){
  const frame=DATA.frames[idx];
  let h=`<h3>Frame ${idx+1} / ${DATA.frames.length}</h3>`;
  h+=`<div class="reason">${esc(frame.file)}</div>`;
  if(idx===0){
    h+=`<div class="op op-none">Initial graph produced by the manager from the task + first VM screenshot.</div>`;
  }
  // find the applied replan that corresponds to this transition (idx-th applied)
  const applied=DATA.replans.filter(r=>r.applied&&r.decision);
  if(idx>0 && applied[idx-1]){
    const r=applied[idx-1];
    const d=r.decision;
    h+=`<div class="reason">replan #${r.iteration} · trigger: ${esc(r.trigger_sid)} · ${esc(r.reason)}</div>`;
    // WHY (1): what the triggering subtask reported (the proximate cause)
    const tstat=DATA.status[r.trigger_sid]||'?';
    const tresp=DATA.responses[r.trigger_sid]||'';
    if(tresp){
      const tcol=tstat==='success'?'#7ee2a8':(tstat==='error'?'#ffa0b0':'#9aa7b5');
      h+=`<div class="op op-rm"><b>why:</b> trigger <b>${esc(r.trigger_sid)}</b> `+
         `→ <span style="color:${tcol}">[${esc(tstat)}]</span> ${esc(trunc(tresp,140))}</div>`;
    }
    // WHY (2): manager's full reasoning (no truncation)
    if(d.reasoning) h+=`<div class="reason"><b>manager reasoning:</b> “${esc(d.reasoning)}”</div>`;
    (d.add||[]).forEach(a=>{h+=`<div class="op op-add">+ add <b>${esc(a.id)}</b> <span style="color:#9aa7b5">(retry of ${esc(r.trigger_sid)})</span>${a.variant_of?` (variant of ${esc(a.variant_of)})`:''}<br>${esc(trunc(a.description,90))}</div>`;});
    (d.remove||[]).forEach(x=>{h+=`<div class="op op-rm">− remove <b>${esc(typeof x==='object'?x.id:x)}</b></div>`;});
    (d.cancel||[]).forEach(x=>{h+=`<div class="op op-rm">✗ cancel <b>${esc(typeof x==='object'?x.id:x)}</b></div>`;});
    Object.entries(d.modify||{}).forEach(([nid,ch])=>{
      let parts=[];
      if(ch.dependencies) parts.push('deps→['+ch.dependencies.map(esc).join(', ')+']');
      if(ch.instruction) parts.push('instruction rewritten');
      h+=`<div class="op op-mod">~ modify <b>${esc(nid)}</b>: ${parts.join(', ')}</div>`;
    });
  }
  // node list with statuses + FULL instruction (not truncated) per node
  h+=`<h3>Nodes in this frame</h3>`;
  frame.nodes.forEach(n=>{
    const s=effStatus(n, idx);
    const col=s==='success'?'#7ee2a8':(s==='error'?'#ffa0b0':(s==='pending'?'#ffd97a':'#9aa7b5'));
    h+=`<div style="margin:6px 0 2px"><span class="dot" style="background:${colorOf(s)}"></span>`+
       `<b>${esc(n.id)}</b> <span style="color:${col}">[${esc(s)}]</span>`+
       (n.deps.length?` <span style="color:#7a8694">← ${n.deps.map(esc).join(', ')}</span>`:'')+`</div>`;
    if(n.desc) h+=`<div style="color:#9aa7b5;font-size:12px;margin-left:14px">${esc(n.desc)}</div>`;
    if(n.instr){
      h+=`<details style="margin:2px 0 6px 14px"><summary style="cursor:pointer;color:#7fd0ff;font-size:12px">full instruction</summary>`+
         `<div class="instr">${esc(n.instr)}</div></details>`;
    }
  });
  if(idx===DATA.frames.length-1 && DATA.final_response){
    h+=`<h3>Final response</h3><div class="resp">${esc(DATA.final_response)}</div>`;
  }
  return h;
}

function render(){
  document.getElementById('graph').innerHTML=renderGraph(DATA.frames[cur], cur);
  document.getElementById('side').innerHTML=renderSide(cur);
  const f=DATA.frames[cur];
  document.getElementById('frameLabel').textContent=
    `[${cur+1}/${DATA.frames.length}] ${f.label}`;
  document.getElementById('prev').disabled=(cur===0);
  document.getElementById('next').disabled=(cur===DATA.frames.length-1);
}
document.getElementById('prev').onclick=()=>{if(cur>0){cur--;render();}};
document.getElementById('next').onclick=()=>{if(cur<DATA.frames.length-1){cur++;render();}};
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')document.getElementById('prev').click();
  if(e.key==='ArrowRight')document.getElementById('next').click();
});
render();
</script>
</body></html>"""


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a single MACU task run dir.")
    ap.add_argument("task_dir", help="Path to one task folder, e.g. runs/<run>/<task_id>")
    ap.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    ap.add_argument("--open", action="store_true", help="Print the HTML path as a file:// URL")
    args = ap.parse_args()

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"Not a directory: {task_dir}")
    if not (task_dir / "final_results.json").exists() and not (task_dir / "graph_snapshots").exists():
        raise SystemExit(
            f"{task_dir} doesn't look like a MACU task dir "
            "(no final_results.json or graph_snapshots/). "
            "Pass a single task folder, not the run root.")

    print_report(task_dir, use_color=not args.no_color)

    if not args.no_html:
        out = task_dir / "analysis.html"
        out.write_text(build_html(task_dir), encoding="utf-8")
        print(f"\n  HTML visualization → {out}")
        if args.open:
            print(f"  open: file://{out}")


if __name__ == "__main__":
    main()
