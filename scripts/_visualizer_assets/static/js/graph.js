/* Dependency-graph SVG renderer for the MACU example animation.
 *
 * Builds a topologically-laid-out DAG and assigns one of:
 *   pending     — node hasn't started
 *   running     — node is the currently-active subagent
 *   done        — completed and on the path to the aggregator
 *   done_offpath— completed but the final answer ignored it
 *   cancelled   — replanned away by the manager
 *
 * Colors mirror those in the paper figures.
 */

const { useMemo } = React;

// Palette matches Fig. 1 of the paper:
//   • Pending  → white fill, slate-grey stroke    #B0BCC8
//   • Done     → pale green fill, green stroke    #2F9C52
//   • Running  → white fill, brick-red stroke     #B23A2E
//   • Off-path → pale grey-blue, slate stroke
//   • Cancelled→ near-white, dashed grey
// The aggregator uses the SAME palette as CUA subagents (just labelled
// differently in its header pill).
const STATUS = {
  pending:      { fill: '#FFFFFF', stroke: '#B0BCC8', text: '#5E6C80' },
  running:      { fill: '#FFFFFF', stroke: '#B23A2E', text: '#0F1F3D' },
  done:         { fill: '#E7F4EA', stroke: '#2F9C52', text: '#1E6432' },
  done_offpath: { fill: '#F0F3F6', stroke: '#546E7A', text: '#1C2A3A' },
  cancelled:    { fill: '#FAFBFC', stroke: '#B0BCC8', text: '#98A3B3' },
};
const MANAGER = STATUS;

// Topological "longest path from sources" layer assignment.
function topoLayers(nodes, edges) {
  const ids = nodes.map(n => n.id);
  const inEdges = {}, outEdges = {};
  for (const id of ids) { inEdges[id] = []; outEdges[id] = []; }
  for (const e of edges) {
    inEdges[e.dst].push(e.src);
    outEdges[e.src].push(e.dst);
  }
  const layer = {};
  const queue = ids.filter(id => inEdges[id].length === 0);
  for (const id of queue) layer[id] = 0;
  const indeg = {};
  for (const id of ids) indeg[id] = inEdges[id].length;
  let head = 0;
  while (head < queue.length) {
    const u = queue[head++];
    for (const v of outEdges[u]) {
      layer[v] = Math.max(layer[v] ?? 0, (layer[u] ?? 0) + 1);
      indeg[v]--;
      if (indeg[v] === 0) queue.push(v);
    }
  }
  const layers = {};
  for (const id of ids) {
    const L = layer[id] ?? 0;
    (layers[L] = layers[L] || []).push(id);
  }
  return Object.keys(layers).sort((a, b) => +a - +b).map(k => layers[k]);
}

function buildGraph(snapshot) {
  const subs = snapshot.subtasks;
  const nodes = subs.map(s => ({ id: s.id, isAgg: false, raw: s }));
  if (snapshot.aggregation) {
    nodes.push({ id: snapshot.aggregation.id, isAgg: true, raw: snapshot.aggregation });
  }
  const edges = [];
  const seen = new Set();
  for (const s of subs) {
    for (const d of s.dependencies || []) {
      if (subs.find(x => x.id === d) || (snapshot.aggregation && snapshot.aggregation.id === d)) {
        const k = `${d}→${s.id}`;
        if (!seen.has(k)) { seen.add(k); edges.push({ src: d, dst: s.id }); }
      }
    }
  }
  if (snapshot.aggregation) {
    for (const d of snapshot.aggregation.dependencies || []) {
      if (subs.find(x => x.id === d)) {
        const k = `${d}→${snapshot.aggregation.id}`;
        if (!seen.has(k)) { seen.add(k); edges.push({ src: d, dst: snapshot.aggregation.id }); }
      }
    }
  }
  // Transitive reduction for aggregator edges: drop u→agg if u→v→agg exists.
  if (snapshot.aggregation) {
    const aggId = snapshot.aggregation.id;
    const adj = {};
    for (const n of nodes) adj[n.id] = [];
    for (const e of edges) if (e.dst !== aggId) adj[e.src].push(e.dst);
    const reaches = (a, b) => {
      const stack = [a]; const seen2 = new Set();
      while (stack.length) {
        const x = stack.pop();
        if (x === b) return true;
        if (seen2.has(x)) continue;
        seen2.add(x);
        for (const n of adj[x] || []) stack.push(n);
      }
      return false;
    };
    const preds = edges.filter(e => e.dst === aggId).map(e => e.src);
    const toRemove = new Set();
    for (const u of preds) for (const v of preds) {
      if (u !== v && reaches(u, v)) { toRemove.add(`${u}→${aggId}`); break; }
    }
    for (let i = edges.length - 1; i >= 0; i--) {
      if (toRemove.has(`${edges[i].src}→${edges[i].dst}`)) edges.splice(i, 1);
    }
  }
  return { nodes, edges, aggId: snapshot.aggregation?.id ?? null };
}

function layoutGraph(graph, masterOrder, opts = {}) {
  const { nodeW = 158, nodeH = 124, layerGap = 162, colGap = 20 } = opts;
  const layers = topoLayers(graph.nodes, graph.edges);
  const orderIdx = {};
  masterOrder.forEach((id, i) => orderIdx[id] = i);
  const pos = {};
  let maxLayerWidth = 0;
  layers.forEach((gen, layerIdx) => {
    gen.sort((a, b) => (orderIdx[a] ?? 999) - (orderIdx[b] ?? 999) || a.localeCompare(b));
    const w = gen.length * (nodeW + colGap) - colGap;
    maxLayerWidth = Math.max(maxLayerWidth, w);
    gen.forEach((id, col) => {
      const x = col * (nodeW + colGap) + nodeW / 2 - w / 2;
      const y = layerIdx * layerGap + nodeH / 2;
      pos[id] = { x, y, layer: layerIdx, col };
    });
  });
  for (const id in pos) pos[id].x += maxLayerWidth / 2;
  return {
    pos, layers, totalW: maxLayerWidth,
    totalH: (layers.length - 1) * layerGap + nodeH,
    nodeW, nodeH, layerGap, colGap,
  };
}

function edgePath(p1, p2, sideHint = 0, layerSpan = 1) {
  const dy = p2.y - p1.y;
  const cy1 = p1.y + dy * 0.5;
  const cy2 = p2.y - dy * 0.5;
  let cx1 = p1.x, cx2 = p2.x;
  if (layerSpan > 1) {
    const bend = (28 + 10 * (layerSpan - 1)) * (sideHint || (p1.x >= p2.x ? 1 : -1));
    cx1 += bend;
  }
  return `M ${p1.x} ${p1.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${p2.x} ${p2.y}`;
}

function GraphNode({ x, y, w, h, label, isAgg, status, showHalo, agentLabel, onClick, isSelected, thumbnailSrc, thumbnailNote, nodeId }) {
  const palette = isAgg ? MANAGER : STATUS;
  const tok = palette[status] || palette.pending || STATUS.pending;
  const lines = label.split('\n');
  const clickable = !!onClick;
  const clipId = `clip-${(nodeId || '').replace(/[^a-z0-9]/gi, '_')}`;
  // Layout inside the node: header strip (badge row) + label + thumbnail.
  // Heights are tuned so the label has consistent space regardless of
  // whether a thumbnail is present.
  const HEADER_H = 18;
  const LABEL_H = 30;
  const PAD_X = 8;
  const PAD_BOTTOM = 8;
  const thumbX = PAD_X;
  const thumbY = HEADER_H + LABEL_H + 2;
  const thumbW = w - PAD_X * 2;
  const thumbH = h - thumbY - PAD_BOTTOM;
  return (
    <g
      transform={`translate(${x - w / 2}, ${y - h / 2})`}
      onClick={onClick}
      style={{ cursor: clickable ? 'pointer' : 'default' }}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } } : undefined}
    >
      {showHalo && (
        <rect
          x={-6} y={-6} width={w + 12} height={h + 12}
          rx={4} ry={4}
          fill="rgba(178,58,46,0.08)"
          stroke="#B23A2E"
          strokeWidth={1.5}
          strokeDasharray="4 2"
        >
          <animate attributeName="opacity" values="0.55;1;0.55" dur="2.4s" repeatCount="indefinite" />
        </rect>
      )}
      {/* Selected outline: blue ring around the active node so the user
          can see which DAG node corresponds to the panel on the right.
          vectorEffect=non-scaling-stroke keeps the ring exactly 3 screen
          pixels wide regardless of the SVG's viewBox-to-pixel ratio, so
          the auto-selected initial node and a click-selected node look
          identical. */}
      {isSelected && (
        <rect
          x={-5} y={-5} width={w + 10} height={h + 10}
          rx={6} ry={6}
          fill="none"
          stroke="#2563EB"
          strokeWidth={3}
          vectorEffect="non-scaling-stroke"
          pointerEvents="none"
        />
      )}
      <rect
        x={0} y={0} width={w} height={h}
        rx={3} ry={3}
        fill={tok.fill}
        stroke={tok.stroke}
        strokeWidth={isAgg ? 2 : 1.6}
        strokeDasharray={status === 'cancelled' ? '4 3' : '0'}
      />
      {/* Header strip + label (above the thumbnail) */}
      <foreignObject x={0} y={0} width={w} height={HEADER_H + LABEL_H}>
        <div xmlns="http://www.w3.org/1999/xhtml" style={{
          width: '100%', height: '100%', boxSizing: 'border-box',
          padding: `5px ${PAD_X}px 0`,
          display: 'flex', flexDirection: 'column', gap: 1,
          fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
          color: tok.text,
          opacity: status === 'cancelled' ? 0.55 : 1,
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            fontSize: 8.5, letterSpacing: '0.08em', textTransform: 'uppercase',
            fontWeight: 700, color: tok.stroke, opacity: 0.95,
            height: 12,
          }}>
            <span>{isAgg ? 'Aggregator' : agentLabel}</span>
            <span style={{ display: 'inline-flex' }}>
              {status === 'done' && (
                <svg viewBox="0 0 16 16" width={11} height={11} aria-hidden="true">
                  <path d="M3 8.5l3 3 6.5-7" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
              {status === 'running' && (
                <svg viewBox="0 0 16 16" width={11} height={11} aria-hidden="true">
                  <circle cx={8} cy={8} r={3} fill="currentColor">
                    <animate attributeName="r" values="3;5;3" dur="1.4s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="1;0.5;1" dur="1.4s" repeatCount="indefinite" />
                  </circle>
                </svg>
              )}
              {status === 'cancelled' && (
                <svg viewBox="0 0 16 16" width={11} height={11} aria-hidden="true">
                  <circle cx={8} cy={8} r={5} fill="none" stroke="currentColor" strokeWidth={1.4} />
                  <line x1={4} y1={12} x2={12} y2={4} stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" />
                </svg>
              )}
              {status === 'done_offpath' && (
                <svg viewBox="0 0 16 16" width={11} height={11} aria-hidden="true">
                  <path d="M3 8.5l3 3 6.5-7" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round" opacity={0.6} />
                </svg>
              )}
            </span>
          </div>
          <div style={{
            fontSize: 10.5, fontWeight: 600, lineHeight: 1.15,
            color: tok.text, whiteSpace: 'pre-wrap',
            textDecoration: status === 'cancelled' ? 'line-through' : 'none',
            marginTop: 2,
          }}>
            {label}
          </div>
        </div>
      </foreignObject>

      {/* Live screenshot preview — fills the rest of the node. For pending
          nodes we show a hairline placeholder so the layout doesn't jitter. */}
      {thumbH > 12 && (
        thumbnailSrc ? (
          <g>
            <clipPath id={clipId}>
              <rect x={thumbX} y={thumbY} width={thumbW} height={thumbH} rx={2} ry={2} />
            </clipPath>
            <image
              href={thumbnailSrc}
              x={thumbX} y={thumbY}
              width={thumbW} height={thumbH}
              preserveAspectRatio="xMidYMid slice"
              clipPath={`url(#${clipId})`}
              opacity={status === 'cancelled' || status === 'done_offpath' ? 0.45 : 1}
            />
            <rect
              x={thumbX} y={thumbY} width={thumbW} height={thumbH}
              rx={2} ry={2}
              fill="none"
              stroke={tok.stroke}
              strokeOpacity={0.35}
              strokeWidth={0.8}
            />
          </g>
        ) : (
          <g>
            <rect
              x={thumbX} y={thumbY} width={thumbW} height={thumbH}
              rx={2} ry={2}
              fill="#F4F6F9"
              stroke={tok.stroke}
              strokeOpacity={0.35}
              strokeWidth={0.8}
              strokeDasharray="2 2"
            />
            <text
              x={thumbX + thumbW / 2} y={thumbY + thumbH / 2 + 3}
              textAnchor="middle"
              style={{
                fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
                fontSize: 8, fill: '#98A3B3', letterSpacing: '0.08em',
                textTransform: 'uppercase', fontWeight: 600,
              }}
            >
              {thumbnailNote || 'no preview yet'}
            </text>
          </g>
        )
      )}
    </g>
  );
}

window.DepGraphRenderer = function DepGraphRenderer({ snapshot, statusFor, newSet, masterOrder, shortLabels, onNodeClick, selectedId, thumbnailFor }) {
  const graph = useMemo(() => buildGraph(snapshot), [snapshot]);
  const layout = useMemo(
    () => layoutGraph(graph, masterOrder || graph.nodes.map(n => n.id)),
    [graph, masterOrder]
  );
  const padX = 16, padY = 16;
  const contentW = layout.totalW + padX * 2;
  const contentH = layout.totalH + padY * 2;
  const xs = Object.values(layout.pos).map(p => p.x);
  const xMid = (Math.min(...xs) + Math.max(...xs)) / 2;

  // Cap the rendered node width at ~170px. The viewBox node width is fixed
  // at layout.nodeW (158), so when there are very few nodes (e.g. OSWorld
  // tasks with 1 subtask) the SVG would otherwise scale way up to fill the
  // stage. Compute a max width that keeps node display size in check.
  const MAX_NODE_PX = 170;
  const scaleCap = MAX_NODE_PX / layout.nodeW;
  const maxW = Math.round(contentW * scaleCap);
  const maxH = Math.round(contentH * scaleCap);

  return (
    <svg
      viewBox={`0 0 ${contentW} ${contentH}`}
      width="100%" height="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ display: 'block', maxWidth: maxW, maxHeight: maxH }}
    >
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0 0 L10 5 L0 10 z" fill="#324A6E" />
        </marker>
        <marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0 0 L10 5 L0 10 z" fill="#B0BCC8" />
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0 0 L10 5 L0 10 z" fill="#B23A2E" />
        </marker>
      </defs>
      <g transform={`translate(${padX}, ${padY})`}>
        {/* edges first */}
        {graph.edges.map((e, i) => {
          const sp = layout.pos[e.src];
          const dp = layout.pos[e.dst];
          if (!sp || !dp) return null;
          const sFrom = { x: sp.x, y: sp.y + layout.nodeH / 2 };
          const sTo = { x: dp.x, y: dp.y - layout.nodeH / 2 };
          const layerSpan = Math.max(1, Math.abs(dp.layer - sp.layer));
          const sideHint = layerSpan > 1 ? (sp.x > xMid ? 1 : -1) : 0;
          const stSrc = statusFor(e.src);
          const stDst = statusFor(e.dst);
          const isActive = stSrc === 'running' || stDst === 'running';
          const muted = stSrc === 'cancelled' || stDst === 'cancelled' || stDst === 'pending' && !isActive;
          let stroke = muted ? '#B0BCC8' : '#324A6E';
          let strokeWidth = muted ? 1.3 : 1.6;
          let marker = muted ? 'url(#arrow-muted)' : 'url(#arrow)';
          if (isActive) {
            stroke = '#B23A2E';
            strokeWidth = 1.8;
            marker = 'url(#arrow-active)';
          }
          return (
            <path
              key={i}
              d={edgePath(sFrom, sTo, sideHint, layerSpan)}
              fill="none"
              stroke={stroke}
              strokeWidth={strokeWidth}
              strokeDasharray={stSrc === 'cancelled' || stDst === 'cancelled' ? '4 3' : '0'}
              markerEnd={marker}
            />
          );
        })}
        {/* nodes */}
        {graph.nodes.map(n => {
          const p = layout.pos[n.id];
          if (!p) return null;
          const status = statusFor(n.id);
          const label = (shortLabels && shortLabels[n.id]) || n.id.replace(/_/g, ' ');
          const showHalo = newSet && newSet.has(n.id);
          const thumb = thumbnailFor ? thumbnailFor(n.id, status) : null;
          return (
            <GraphNode
              key={n.id}
              nodeId={n.id}
              x={p.x} y={p.y}
              w={layout.nodeW} h={layout.nodeH}
              label={label}
              isAgg={n.isAgg}
              status={status}
              showHalo={showHalo}
              agentLabel="CUA Subagent"
              onClick={onNodeClick ? () => onNodeClick(n.id) : undefined}
              isSelected={selectedId === n.id}
              thumbnailSrc={thumb?.src}
              thumbnailNote={thumb?.note}
            />
          );
        })}
      </g>
    </svg>
  );
};

// Compute set of nodes on the path to aggregator (those whose work reaches
// the final answer). Used to demote off-path completed nodes to grey.
window.computePathToAgg = function computePathToAgg(snapshot) {
  if (!snapshot.aggregation) return new Set();
  const aggId = snapshot.aggregation.id;
  const adj = {};  // node -> list of consumers
  const allIds = new Set([aggId]);
  for (const s of snapshot.subtasks) {
    allIds.add(s.id);
    for (const d of s.dependencies || []) {
      (adj[d] = adj[d] || []).push(s.id);
    }
  }
  for (const d of snapshot.aggregation.dependencies || []) {
    (adj[d] = adj[d] || []).push(aggId);
  }
  const memo = new Map();
  const reaches = (n) => {
    if (n === aggId) return true;
    if (memo.has(n)) return memo.get(n);
    memo.set(n, false);
    for (const next of adj[n] || []) {
      if (reaches(next)) { memo.set(n, true); return true; }
    }
    return false;
  };
  const out = new Set();
  for (const id of allIds) if (reaches(id)) out.add(id);
  return out;
};

// Union of node ids across all snapshots, in topo order — keeps node columns
// stable across the animation.
window.unionMasterOrder = function unionMasterOrder(snapshots) {
  const seen = [], seenSet = new Set();
  for (const s of snapshots) {
    const g = buildGraph(s);
    const layers = topoLayers(g.nodes, g.edges);
    for (const gen of layers) {
      for (const id of [...gen].sort()) {
        if (!seenSet.has(id)) { seen.push(id); seenSet.add(id); }
      }
    }
  }
  return seen;
};
