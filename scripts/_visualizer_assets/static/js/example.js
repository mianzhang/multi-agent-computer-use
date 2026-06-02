/* The animated example player.
 *
 * Drives:
 *   - a "wall clock" that increments at speed×realtime
 *   - the dependency graph (renders the snapshot active at the current t)
 *   - a tab strip across the right side that shows each subtask's progress
 *     (the currently-running one is auto-selected unless the user clicks)
 *   - a screenshot panel that shows the keyframe at the current t for the
 *     selected tab
 *
 * Time mapping: real task is ~4500s of wall clock; we play it back in ~45s
 * at 1× speed, with 2× and 4× options.
 */

const { useEffect, useMemo, useRef, useState, useCallback } = React;

const REAL_TO_PLAYBACK = 100;  // 1 sec playback = 100 sec real time

function truncateWords(text, n) {
  if (!text) return '';
  const words = text.trim().split(/\s+/);
  if (words.length <= n) return text;
  return words.slice(0, n).join(' ') + '…';
}

function fmtClock(realSec) {
  const m = Math.floor(realSec / 60);
  const s = Math.floor(realSec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// Per-task text customisations. Each entry tailors the captions, marker
// labels, and chrome URLs to a specific example so the player reads as a
// narrated walkthrough rather than a generic timeline.
// Standalone visualizer: no per-task hand-curated captions. Every hook below
// short-circuits when the table is empty, so the player falls back to:
// marker.short = snapshot.label, chromeUrl = "—", generic captions.
const TASK_OVERRIDES = {};

function overridesFor(taskId) {
  const key = Object.keys(TASK_OVERRIDES).find(k => taskId.startsWith(k));
  return key ? TASK_OVERRIDES[key] : null;
}

// Build the host URL that should appear in the chrome bar of a screenshot
// frame — derived from the subtask id so each tab feels like a real browser.
function chromeUrlFor(subtaskId, taskId) {
  const o = overridesFor(taskId);
  return (o && o.chromeUrl(subtaskId)) || '—';
}

// Pick the active snapshot for time t: the most recent snapshot whose t <= now.
function activeSnapshotIdx(snapshots, t) {
  let idx = 0;
  for (let i = 0; i < snapshots.length; i++) {
    if (snapshots[i].t <= t) idx = i;
  }
  return idx;
}

// Which keyframe (index in the subtask's frame list) should be shown
// for a subtask at simulated time t. We interpolate linearly between start
// and end of the subtask.
function frameAtTime(sched, frames, t) {
  if (!frames || frames.length === 0) return null;
  if (t <= sched.start) return 0;
  if (t >= sched.end) return frames.length - 1;
  const p = (t - sched.start) / Math.max(1, sched.end - sched.start);
  return Math.min(frames.length - 1, Math.floor(p * frames.length));
}

// Compute statuses for a subtask at simulated time t:
//   pending  — t < schedule.start
//   running  — schedule.start <= t < schedule.end
//   done     — t >= schedule.end (or done_offpath/cancelled by manager state)
function rawStatusAt(sched, t) {
  if (!sched) return 'pending';
  if (t < sched.start) return 'pending';
  if (t < sched.end) return 'running';
  return 'done';
}

// Cancelled set per snapshot: nodes that appear in this snapshot but were
// dropped by the final snapshot — i.e., replanned away. The original
// subtask_beach_chalet completes successfully but its dependency is later
// rewired to subtask_beach_chalet_hours_retry; treat it as off-path / done.
function cancelledInFinal(currentSnap, finalSnap) {
  const finalIds = new Set();
  for (const s of finalSnap.subtasks) finalIds.add(s.id);
  if (finalSnap.aggregation) finalIds.add(finalSnap.aggregation.id);
  // currentSnap nodes not in finalSnap → cancelled (we have none here)
  // and finalSnap nodes whose agg-deps no longer include currentSnap → off-path.
  return new Set();
}

window.ExamplePlayer = function ExamplePlayer({ data }) {
  const totalReal = data.total_seconds;
  const [tReal, setTReal] = useState(0);             // current sim time (real seconds)
  // Start paused — we resume once the player scrolls into view (or, after
  // a tab switch, immediately, since the player is already in view).
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);             // 1x, 2x, 4x
  const [selectedTab, setSelectedTab] = useState(null); // null = follow active
  const [taskOpen, setTaskOpen] = useState(false);       // task-text "More" toggle

  const masterOrder = useMemo(() => window.unionMasterOrder(data.snapshots), [data.snapshots]);

  const scheduleById = useMemo(() => {
    const m = {};
    for (const s of data.schedule) m[s.id] = s;
    return m;
  }, [data.schedule]);

  const subtaskById = useMemo(() => {
    const m = {};
    for (const s of data.subtasks) m[s.id] = s;
    return m;
  }, [data.subtasks]);

  // RAF-driven clock
  const rafRef = useRef(null);
  const lastTickRef = useRef(null);

  useEffect(() => {
    if (!playing) {
      lastTickRef.current = null;
      return;
    }
    const tick = (now) => {
      if (lastTickRef.current == null) lastTickRef.current = now;
      const dtMs = now - lastTickRef.current;
      lastTickRef.current = now;
      setTReal((t) => {
        const next = t + (dtMs / 1000) * REAL_TO_PLAYBACK * speed;
        if (next >= totalReal) {
          // Stop at end; user can press restart.
          setPlaying(false);
          return totalReal;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, speed, totalReal]);

  // Decide the active snapshot
  const snapIdx = activeSnapshotIdx(data.snapshots, tReal);
  const snapshot = data.snapshots[snapIdx];
  const finalSnapshot = data.snapshots[data.snapshots.length - 1];

  // The replan flash badge — fade in for ~1s after a snapshot change
  const [replanFlash, setReplanFlash] = useState(false);
  const lastSnapIdxRef = useRef(0);
  useEffect(() => {
    if (snapIdx !== lastSnapIdxRef.current && snapIdx > 0) {
      setReplanFlash(true);
      const t = setTimeout(() => setReplanFlash(false), 2500);
      lastSnapIdxRef.current = snapIdx;
      return () => clearTimeout(t);
    }
    lastSnapIdxRef.current = snapIdx;
  }, [snapIdx]);

  // For each node in the current snapshot, compute its status at tReal.
  // Manager / aggregator runs after all upstream are done.
  // pathSet is recomputed per-snapshot so the original Beach Chalet node
  // (which the manager later drops from the aggregator's deps) demotes to
  // 'done_offpath' precisely when its role in the DAG is replaced.
  const currentPathSet = useMemo(() => window.computePathToAgg(snapshot), [snapshot]);
  const statusFor = useCallback((nodeId) => {
    if (nodeId === finalSnapshot.aggregation?.id) {
      // aggregator finishes when all its deps in the CURRENT snapshot are done
      const deps = snapshot.aggregation?.dependencies || [];
      const allDone = deps.every(d => {
        const sched = scheduleById[d];
        return sched && tReal >= sched.end;
      });
      if (allDone && tReal >= totalReal - 20) return 'done';
      if (allDone) return 'running';
      return 'pending';
    }
    const sched = scheduleById[nodeId];
    const raw = rawStatusAt(sched, tReal);
    if (raw === 'done') {
      // 'done' but not on path to aggregator (in this snapshot) → off-path grey.
      if (currentPathSet.size && !currentPathSet.has(nodeId)) return 'done_offpath';
      return 'done';
    }
    return raw;
  }, [tReal, scheduleById, snapshot, currentPathSet, finalSnapshot, totalReal]);

  // newSet — nodes that just appeared in the current snapshot vs previous.
  // Halos them for ~3s after the replan.
  const newSet = useMemo(() => {
    if (snapIdx === 0) return new Set();
    const prev = data.snapshots[snapIdx - 1];
    const prevIds = new Set();
    for (const s of prev.subtasks) prevIds.add(s.id);
    if (prev.aggregation) prevIds.add(prev.aggregation.id);
    const out = new Set();
    for (const s of snapshot.subtasks) if (!prevIds.has(s.id)) out.add(s.id);
    if (snapshot.aggregation && !prevIds.has(snapshot.aggregation.id)) out.add(snapshot.aggregation.id);
    // Fade out the halo over time (only show in first ~5 sim minutes after replan)
    const timeSinceSnap = tReal - snapshot.t;
    if (timeSinceSnap > 600) return new Set();
    return out;
  }, [snapIdx, snapshot, tReal, data.snapshots]);

  // Aggregator is special — it appears as a tab once all its upstream
  // subtasks have completed, and the right-hand panel renders the final
  // response instead of a browser screenshot.
  const aggId = finalSnapshot.aggregation?.id;
  const aggDeps = finalSnapshot.aggregation?.dependencies || [];
  const aggReady = aggDeps.every(d => {
    const sched = scheduleById[d];
    return sched && tReal >= sched.end;
  });

  // Tabs mirror the DAG: every subtask that exists in the current snapshot
  // gets a tab, plus the aggregator. Pending nodes are clickable but render
  // a "not started yet" panel. Keeping the strip stable across the run also
  // means the user can pre-select a node and watch its progress come in.
  const tabsToShow = useMemo(() => {
    const ordered = data.subtask_order.filter(id =>
      snapshot.subtasks.some(x => x.id === id)
    );
    if (aggId) ordered.push(aggId);
    return ordered;
  }, [data.subtask_order, snapshot, aggId]);

  // Determine "active" tab: when the aggregator is ready, it's always active.
  // Otherwise, the bottleneck running subagent (or most-recently-completed).
  const activeTab = useMemo(() => {
    if (aggReady && aggId) return aggId;
    let running = null;
    let lastDone = null;
    for (const id of tabsToShow) {
      const s = scheduleById[id];
      if (!s) continue;
      if (tReal >= s.start && tReal < s.end) {
        if (!running || s.end > scheduleById[running].end) running = id;
      } else if (tReal >= s.end) {
        if (!lastDone || s.end > scheduleById[lastDone].end) lastDone = id;
      }
    }
    return running || lastDone || tabsToShow[0] || null;
  }, [tabsToShow, scheduleById, tReal, aggReady, aggId]);

  // Any node id that's in the current snapshot is a legal selection from
  // the DAG (not just tabs that have started). If the user clicks a pending
  // node, we still surface it on the right with a "not started yet" state.
  const allValidIds = useMemo(() => {
    const s = new Set(snapshot.subtasks.map(x => x.id));
    if (aggId) s.add(aggId);
    return s;
  }, [snapshot, aggId]);
  const currentTab = selectedTab && allValidIds.has(selectedTab) ? selectedTab : activeTab;
  const isAggTab = currentTab === aggId;
  const currentStatus = currentTab ? statusFor(currentTab) : null;

  // Per-task overrides for prose captions and marker labels. Some examples
  // (e.g. 4614) reuse the same snapshot label across distinct events, so we
  // also disambiguate by occurrence index — keys like "after_subtask_x:1".
  const overrides = overridesFor(data.task_id) || {};
  const snapshotKeyMap = useMemo(() => {
    const counts = {};
    return data.snapshots.map(s => {
      const seen = counts[s.label] || 0;
      counts[s.label] = seen + 1;
      return `${s.label}:${seen}`;
    });
  }, [data.snapshots]);
  function snapshotOverride(table, snap, idx) {
    if (!table) return null;
    return table[snapshotKeyMap[idx]] || table[snap.label] || null;
  }

  // Completion summary line. Use the task's hand-crafted stats string if
  // present; otherwise derive from data (subagent count, replan count, wall
  // clock minutes).
  const completionStats = overrides.completionStats || (() => {
    const subagentCount = (data.subtask_order || []).length;
    const replanCount = Math.max(0, data.snapshots.length - 1);
    const wallclockMin = Math.round((data.wallclock_total_seconds || data.total_seconds || 0) / 60);
    return `${wallclockMin} min wall clock · ${subagentCount} subagents · ${replanCount} replans · 1 aggregated response`;
  })();

  // Caption: short description of what's happening right now.
  // Captions cover three cases (in priority order): (1) a snapshot transition
  // just fired — describe the replan; (2) the aggregator is running/done;
  // (3) the active CUA subagent's progress.
  const caption = useMemo(() => {
    if (tReal >= totalReal - 1) {
      return (
        <span><span className="player__caption-pill">✓ Task complete</span> {completionStats}. <button type="button" className="player__caption-link" onClick={() => { setTReal(0); setPlaying(true); setSelectedTab(null); }}>Restart</button> to watch again.</span>
      );
    }
    // Replan flash wins over the aggregator caption — when a replan fires
    // right at the boundary where the aggregator begins (e.g. 4614's retry
    // completion), users should still see the replan story first.
    if (replanFlash && snapIdx > 0) {
      const t = snapshotOverride(overrides.replanText, snapshot, snapIdx);
      if (t) return t;
      return <span><strong>Replan.</strong> The manager updated the DAG based on the latest subagent response.</span>;
    }
    if (tReal >= totalReal - 30) {
      return overrides.aggregatorCaption || (
        <span><strong>Aggregator (manager).</strong> Synthesises all subagent responses into a single decision summary. See the response on the right.</span>
      );
    }
    if (!currentTab) {
      return overrides.initialCaption || <span><strong>Initial decomposition.</strong> The manager breaks the task into parallel CUA subagents and a downstream aggregator.</span>;
    }
    const st = subtaskById[currentTab];
    const sched = scheduleById[currentTab];
    const status = statusFor(currentTab);
    if (status === 'running') {
      const pct = Math.round(((tReal - sched.start) / Math.max(1, sched.end - sched.start)) * 100);
      return <span><strong>{labelOneLine(st.label)}</strong> — CUA subagent working <span className="muted">({pct}% complete)</span></span>;
    }
    if (status === 'done') {
      return <span><strong>{labelOneLine(st.label)}</strong> — subagent completed, manager reviewing the response and any saved files.</span>;
    }
    if (status === 'done_offpath') {
      return <span><strong>{labelOneLine(st.label)}</strong> — completed, but the manager later replanned this out of the aggregator's deps.</span>;
    }
    return <span><strong>{labelOneLine(st.label)}</strong></span>;
  }, [currentTab, statusFor, subtaskById, scheduleById, tReal, totalReal, replanFlash, snapIdx, snapshot]);

  function labelOneLine(label) {
    return label.replace(/\n/g, ' ');
  }

  // When the user interacts with the timeline, a node, or a tab, we pause
  // briefly so they can read the new context before playback resumes. Repeat
  // clicks reset the timer.
  const pauseTimerRef = useRef(null);
  const pauseAndResume = useCallback(() => {
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
    setPlaying(false);
    pauseTimerRef.current = setTimeout(() => {
      setPlaying(true);
      pauseTimerRef.current = null;
    }, 500);
  }, []);
  // Clean up any pending auto-resume on unmount.
  useEffect(() => () => {
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
  }, []);
  // Sync playback to stage visibility:
  //   stage enters viewport → 500ms grace then play
  //   stage leaves viewport → pause (no point playing what nobody's watching)
  // The observer stays connected across both directions so this also handles
  // the user scrolling back and forth.
  const stageRef = useRef(null);
  useEffect(() => {
    const el = stageRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      // Fallback: just start with the usual grace pause.
      pauseAndResume();
      return;
    }
    const obs = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          pauseAndResume();
        } else {
          if (pauseTimerRef.current) { clearTimeout(pauseTimerRef.current); pauseTimerRef.current = null; }
          setPlaying(false);
        }
      }
    }, { threshold: 0.1 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Restart, play/pause handlers
  const onRestart = () => {
    if (pauseTimerRef.current) { clearTimeout(pauseTimerRef.current); pauseTimerRef.current = null; }
    setTReal(0); setPlaying(true); setSelectedTab(null);
  };
  const onPlayPause = () => {
    // Manual toggle should cancel any auto-resume in flight.
    if (pauseTimerRef.current) { clearTimeout(pauseTimerRef.current); pauseTimerRef.current = null; }
    if (tReal >= totalReal) { setTReal(0); setPlaying(true); return; }
    setPlaying(p => !p);
  };

  // Space bar toggles play/pause when the user isn't typing in a field.
  // Read the latest onPlayPause via a ref so we add/remove the listener
  // once instead of on every tReal tick.
  const onPlayPauseRef = useRef(onPlayPause);
  useEffect(() => { onPlayPauseRef.current = onPlayPause; });
  useEffect(() => {
    const onKey = (e) => {
      if (e.code !== 'Space' && e.key !== ' ') return;
      const target = e.target;
      const tag = target && target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (target && target.isContentEditable)) return;
      e.preventDefault();
      onPlayPauseRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Scrub the playhead by pointer position on a gantt track. Supports both
  // single clicks AND click-and-drag — once the user presses down on a
  // track we capture the pointer and follow it (the way native video
  // scrubbers work), so the mouse can wander outside the track and still
  // drive the timeline.
  const dragRectRef = useRef(null);
  const draggingRef = useRef(false);
  const scrubToClientX = (clientX) => {
    const rect = dragRectRef.current;
    if (!rect) return;
    const x = clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    setTReal(ratio * totalReal);
    pauseAndResume();
  };
  const onTrackPointerDown = (e, subtaskId) => {
    // Only main button starts a drag.
    if (e.button !== undefined && e.button !== 0) return;
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) {}
    dragRectRef.current = e.currentTarget.getBoundingClientRect();
    draggingRef.current = true;
    if (subtaskId) setSelectedTab(subtaskId);
    scrubToClientX(e.clientX);
  };
  const onTrackPointerMove = (e) => {
    if (!draggingRef.current) return;
    scrubToClientX(e.clientX);
  };
  const onTrackPointerUp = () => {
    draggingRef.current = false;
    dragRectRef.current = null;
  };

  // Tab/node selection changes only which subtask is *viewed* — it doesn't
  // change the timeline position — so it shouldn't pause playback. seekTo,
  // by contrast, jumps the playhead and so triggers the brief auto-pause.
  const pickTab = (id) => { setSelectedTab(id); };
  const seekTo = (t) => { setTReal(t); pauseAndResume(); };

  // Progress-bar markers — one per snapshot transition. Labels come from
  // per-task overrides; duplicates in `label` are disambiguated by occurrence
  // (e.g. 4614 fires two `after_subtask_american` events, before and after
  // the retry). When two markers fall too close together (e.g. 4614's
  // initial marker @ t=0 vs Replan #1 @ t=9.4, which are <1% apart on a
  // ~38 min timeline), the later marker's text label is suppressed so the
  // captions don't overlap — but its dot still renders on the track.
  const markers = data.snapshots.map((s, i) => ({
    t: s.t,
    label: s.label,
    meta: snapshotOverride(overrides.markers, s, i) || { kind: 'milestone', short: s.label.replace(/^after_subtask_/, ''), long: '' },
  }));
  // Visibility map for marker text labels. We walk markers in order and hide
  // any whose distance to the last visible marker is below MIN_LABEL_GAP
  // (as a fraction of totalReal).
  const MIN_LABEL_GAP = 0.04;
  const markerLabelVisible = useMemo(() => {
    const vis = markers.map(() => false);
    if (markers.length === 0) return vis;
    vis[0] = true;
    let lastT = markers[0].t;
    for (let i = 1; i < markers.length; i++) {
      if ((markers[i].t - lastT) / Math.max(1, totalReal) >= MIN_LABEL_GAP) {
        vis[i] = true;
        lastT = markers[i].t;
      }
    }
    return vis;
  }, [markers, totalReal]);

  // Current frame for selected tab
  const tabFrameIdx = currentTab ? frameAtTime(scheduleById[currentTab], subtaskById[currentTab]?.frames || [], tReal) : null;
  const tabFrames = currentTab ? subtaskById[currentTab]?.frames || [] : [];
  const currentFrame = (tabFrameIdx != null) ? tabFrames[tabFrameIdx] : null;

  return (
    <div className="player">
      <div className="player__header">
        <span className="player__benchmark">{data.benchmark}</span>
        <div className={`player__task ${taskOpen ? '' : 'player__task-clamp'}`}>{data.task_text}</div>
        <button
          type="button"
          className="player__expand"
          onClick={() => setTaskOpen(o => !o)}
          aria-expanded={taskOpen}
          aria-label={taskOpen ? 'Collapse task description' : 'Expand task description'}
        >
          {taskOpen ? 'Less' : 'More'}
        </button>
      </div>

      <div className="gantt">
        <div className="gantt__rows">
          {data.subtask_order.map((id) => {
            const s = scheduleById[id];
            const sub = subtaskById[id];
            const status = statusFor(id);
            const leftPct = (s.start / totalReal) * 100;
            const widthPct = ((s.end - s.start) / totalReal) * 100;
            const isCurrent = currentTab === id;
            return (
              <div key={id} className={`gantt__row gantt__row--${status} ${isCurrent ? 'active' : ''}`}>
                <div className="gantt__label" onClick={() => pickTab(id)}>{labelOneLine(sub.label)}</div>
                <div
                  className="gantt__track"
                  onPointerDown={(e) => onTrackPointerDown(e, id)}
                  onPointerMove={onTrackPointerMove}
                  onPointerUp={onTrackPointerUp}
                  onPointerCancel={onTrackPointerUp}
                  title="Click and drag to scrub"
                >
                  <div
                    className={`gantt__bar gantt__bar--${status}`}
                    style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  >
                    {status === 'running' && (
                      <div
                        className="gantt__bar-progress"
                        style={{ width: `${((tReal - s.start) / (s.end - s.start)) * 100}%` }}
                      />
                    )}
                  </div>
                </div>
                <div className="gantt__dur">{Math.round((s.end - s.start) / 60)}m</div>
              </div>
            );
          })}
          {/* Manager row — also hosts the replan/milestone indicators along
              the full timeline (the row's track spans 0 → totalReal), so it
              doubles as a labelled progress bar. Click anywhere to scrub;
              click a marker dot to jump to that event. */}
          {aggId && (() => {
            const aggStart = totalReal - 30;
            const isCurrent = currentTab === aggId;
            const aggStatus = aggReady ? (tReal >= totalReal - 1 ? 'agg-done' : 'agg-running') : 'agg-pending';
            return (
              <div className={`gantt__row gantt__row--agg ${isCurrent ? 'active' : ''}`}>
                <div className="gantt__label" onClick={() => pickTab(aggId)}><span className="gantt__agg-pill">MANAGER</span><span className="gantt__agg-label-text">Replan, summarize</span></div>
                <div
                  className="gantt__track gantt__track--agg"
                  onPointerDown={(e) => onTrackPointerDown(e, aggId)}
                  onPointerMove={onTrackPointerMove}
                  onPointerUp={onTrackPointerUp}
                  onPointerCancel={onTrackPointerUp}
                  title="Click and drag to scrub"
                >
                  {/* Marker dots ABOVE the bar */}
                  {markers.map((m, i) => {
                    const reached = tReal >= m.t;
                    return (
                      <div
                        key={`dot-${i}`}
                        className={`mgr-marker mgr-marker--${m.kind} ${reached ? 'is-reached' : ''}`}
                        style={{ left: `${(m.t / totalReal) * 100}%` }}
                        onClick={(e) => { e.stopPropagation(); seekTo(m.t); }}
                        title={`${m.meta.short} · ${m.meta.long}`}
                      >
                        <div className="mgr-marker__stem"></div>
                        <div className="mgr-marker__dot"></div>
                      </div>
                    );
                  })}
                  {/* Manager bar itself, positioned at the actual aggregator
                      slot at the very end of the timeline. */}
                  <div
                    className={`gantt__bar gantt__bar--agg gantt__bar--${aggStatus}`}
                    style={{ left: `${(aggStart / totalReal) * 100}%`, width: `${(30 / totalReal) * 100}%` }}
                  />
                  {/* Text labels BELOW the bar — skipped for markers crowded
                      against an earlier visible marker (their dots still
                      show above). */}
                  {markers.map((m, i) => markerLabelVisible[i] ? (
                    <div
                      key={`lbl-${i}`}
                      className={`mgr-marker-label mgr-marker-label--${m.kind}`}
                      style={{ left: `${(m.t / totalReal) * 100}%` }}
                      onClick={(e) => { e.stopPropagation(); seekTo(m.t); }}
                    >
                      <span className="mgr-marker-label__short">{m.meta.short}</span>
                      <span className="mgr-marker-label__long">{m.meta.long}</span>
                    </div>
                  ) : null)}
                </div>
                <div className="gantt__dur">≪1m</div>
              </div>
            );
          })()}
        </div>
        {/* Playhead offsets: track starts at gantt-padding-left (18) +
            row-padding-left (4) + label (160) + column-gap (10) = 192px,
            and ends 68px from the right edge (18 + 4 + 10 + dur 36). */}
        <div className="gantt__playhead" style={{ left: `calc(192px + (100% - 260px) * ${tReal / totalReal})` }} />
      </div>

      <div className="player__controls">
        <button className="player__btn primary" onClick={onPlayPause} aria-label={playing ? 'Pause' : (tReal >= totalReal - 1 ? 'Replay' : 'Play')}>
          {playing ? (
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>
          ) : tReal >= totalReal - 1 ? (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 1 0 3-6.7" />
              <polyline points="3 4 3 10 9 10" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4l13 8-13 8V4z"/></svg>
          )}
        </button>
        <button className="player__btn" onClick={onRestart} aria-label="Restart" title="Restart from beginning">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <polyline points="11 17 6 12 11 7"/>
            <polyline points="18 17 13 12 18 7"/>
          </svg>
        </button>
        <span className="player__time">
          {fmtClock(tReal)} <span className="muted">/ {fmtClock(totalReal)} wall clock</span>
        </span>
        <span className="player__hint">
          <span className="muted">Click any tab or row to switch view · Playback speed:</span>
        </span>
        <div className="player__speeds">
          {[1, 2, 4].map(s => (
            <button
              key={s}
              className={`player__speed ${speed === s ? 'active' : ''}`}
              onClick={() => setSpeed(s)}
            >{s}×</button>
          ))}
        </div>
      </div>

      <div className="player__caption">{caption}</div>

      <div className="player__stage" ref={stageRef}>
        <div className="stage-graph">
          <div className="stage-graph__label">
            <span>Dependency DAG · snapshot {snapIdx + 1} of {data.snapshots.length}</span>
            <span className={`replan-badge ${replanFlash && snapIdx > 0 ? 'visible' : ''}`}>● Replan</span>
          </div>
          <div className="stage-graph__svg">
            <window.DepGraphRenderer
              snapshot={snapshot}
              statusFor={statusFor}
              newSet={newSet}
              masterOrder={masterOrder}
              shortLabels={data.short_labels}
              selectedId={currentTab}
              onNodeClick={(id) => {
                // Always allow selecting a node from the DAG, even if it
                // hasn't started yet — the right panel handles the
                // "not started" / "off-path" states.
                pickTab(id);
              }}
              thumbnailFor={(nodeId, status) => {
                // Live thumbnail = the current frame for that subtask at tReal.
                // For pending nodes we return null so the placeholder shows.
                // The aggregator has no screenshots; show a tiny "manager" hint.
                if (nodeId === aggId) {
                  if (status === 'pending') return { src: null, note: 'pending' };
                  if (aggReady && tReal >= totalReal - 1) {
                    return { src: null, note: '✓ response ready' };
                  }
                  return { src: null, note: 'synthesising…' };
                }
                const sub = subtaskById[nodeId];
                if (!sub) return null;
                const sched = scheduleById[nodeId];
                const frames = sub.frames || [];
                if (!frames.length) return { src: null, note: 'no preview' };
                if (status === 'pending') return { src: null, note: 'pending' };
                const idx = frameAtTime(sched, frames, tReal);
                return { src: frames[idx]?.src, note: null };
              }}
            />
          </div>
          <div className="stage-graph__legend">
            <span><span className="legend-chip" style={{ background: '#E7F4EA', borderColor: '#2F9C52' }}></span> Done</span>
            <span><span className="legend-chip" style={{ background: '#FFFFFF', borderColor: '#B23A2E' }}></span> Running</span>
            <span><span className="legend-chip" style={{ background: '#F0F3F6', borderColor: '#546E7A' }}></span> Done (discarded)</span>
            <span><span className="legend-chip" style={{ background: '#FFFFFF', borderColor: '#B0BCC8' }}></span> Pending</span>
          </div>
        </div>

        <div className="stage-screens">
          <div className="stage-screens__tabs" role="tablist">
            {tabsToShow.map(id => {
              const isAgg = id === aggId;
              const status = statusFor(id);
              const st = isAgg
                ? { label: 'Aggregator: final response' }
                : subtaskById[id];
              if (!st) return null;
              const dotClass = status === 'running' ? 'running'
                : status === 'done' ? 'done'
                : status === 'done_offpath' ? 'done-offpath' : '';
              return (
                <button
                  key={id}
                  className={`stage-screens__tab ${currentTab === id ? 'active' : ''} ${isAgg ? 'agg' : ''}`}
                  onClick={() => pickTab(id)}
                  role="tab"
                  aria-selected={currentTab === id}
                >
                  <span className={`status-dot ${dotClass}`}></span>
                  <span>{labelOneLine(st.label)}</span>
                </button>
              );
            })}
          </div>
          <div className="stage-screens__viewport">
            {isAggTab ? (
              <div className="stage-screens__frame agg-frame">
                <div className="stage-screens__chrome agg-chrome">
                  <span className="agg-chrome__pill">MANAGER</span>
                  <span className="url">final_aggregation · synthesise + report</span>
                  <span className="step-counter">{aggReady ? (tReal >= totalReal - 1 ? '✓ done' : 'in progress…') : 'pending — waiting on upstream subagents'}</span>
                </div>
                {aggReady ? (
                  <div className="agg-response">
                    {window.renderFinalResponse(data.final_response)}
                  </div>
                ) : (
                  <div className="stage-screens__pending">
                    <div className="stage-screens__pending-icon">⏳</div>
                    <div><strong>Aggregator hasn't run yet.</strong></div>
                    <div className="muted">It will fire once all of its dependencies finish. You can keep watching, or click another node to follow its progress.</div>
                  </div>
                )}
              </div>
            ) : currentFrame ? (
              <div className="stage-screens__frame">
                <div className="stage-screens__chrome">
                  <span className="pill"></span><span className="pill"></span><span className="pill"></span>
                  <span className="url">{chromeUrlFor(currentTab, data.task_id)}</span>
                  <span className="step-counter">step {currentFrame.step} · frame {tabFrameIdx + 1}/{tabFrames.length}</span>
                </div>
                <img
                  className="stage-screens__img"
                  src={currentFrame.src}
                  alt={`${currentTab} step ${currentFrame.step}`}
                  loading="lazy"
                />
                {(currentFrame.reasoning || currentFrame.action) && (
                  <div className="stage-screens__caption">
                    {currentFrame.reasoning && (
                      <div className="stage-screens__caption-row">
                        <span className="stage-screens__caption-label">Reasoning</span>
                        <span className="stage-screens__caption-text">{truncateWords(currentFrame.reasoning, 60)}</span>
                      </div>
                    )}
                    {currentFrame.action && (
                      <div className="stage-screens__caption-row">
                        <span className="stage-screens__caption-label">Action</span>
                        <span className="stage-screens__caption-text stage-screens__caption-text--action">{currentFrame.action}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : currentTab && currentStatus === 'pending' ? (
              <div className="stage-screens__pending">
                <div className="stage-screens__pending-icon">○</div>
                <div><strong>{labelOneLine(subtaskById[currentTab]?.label || currentTab)}</strong> — not started yet.</div>
                <div className="muted">This subtask is waiting on its upstream dependencies in the DAG. The playhead will reach it shortly.</div>
              </div>
            ) : (
              <div className="stage-screens__empty">Subagents have not started yet.</div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
};
