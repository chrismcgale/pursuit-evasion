"""Build a self-contained HTML explorer for how the BT and the policies interact.

``viz.py`` renders what happened. ``trace.py`` records why. This turns the trace
into something you can actually interrogate: scrub to any control step and see,
for the selected agent, every gate branch with its live numbers, which one the
Selector took, what the scripted controller and the policy each wanted at that
instant, and whether the safety filter overruled the result.

The output is one HTML file with the traces inlined — no server, no CDN, no build
step. Open it in a browser, or hand it to someone who does not have the repo.

    uv run pe-explore --game assault --seed 0
    uv run pe-explore --game tag --controllers scripted bt_safe --seed 3

Recording several controllers on the *same seed* is the point: identical starts,
and a dropdown to flip between them, is the clearest way to see the gate change
an outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..env.games import GAME_KEYS
from .trace import record

_TEMPLATE = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>pursuit-evasion — gate explorer</title>
<style>
  :root {
    --bg:#0e1116; --panel:#161b22; --line:#30363d; --fg:#c9d1d9; --dim:#8b949e;
    --scripted:#58a6ff; --rl:#f0883e; --applied:#3fb950; --opp:#f85149;
    --asset:#d29922; --shield:#a371f7; --est:#79c0ff;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:10px 16px; border-bottom:1px solid var(--line);
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:14px; margin:0; font-weight:600; letter-spacing:.02em; }
  select, button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
                   border-radius:5px; padding:4px 9px; font:inherit; cursor:pointer; }
  button:hover, select:hover { border-color:var(--dim); }
  button.on { border-color:var(--applied); color:var(--applied); }
  .grid { display:grid; grid-template-columns:1fr 400px; gap:12px; padding:12px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:7px;
           padding:10px 12px; }
  .panel h2 { font-size:11px; text-transform:uppercase; letter-spacing:.09em;
              color:var(--dim); margin:0 0 8px; font-weight:600; }
  canvas { display:block; width:100%; height:auto; background:#0b0e13; border-radius:5px; }
  /* the arena is square; cap the width so it does not become a giant column */
  #top { max-width:760px; margin:0 auto; }
  .row { display:flex; gap:12px; align-items:center; }
  .stack > * + * { margin-top:12px; }
  /* --- behaviour tree --- */
  .branch { border-left:3px solid var(--line); padding:5px 0 5px 9px; margin-bottom:3px;
            opacity:.62; }
  .branch.fired { opacity:1; border-left-color:var(--applied); background:#1c2430;
                  border-radius:0 5px 5px 0; }
  .branch.unreached { opacity:.28; }
  .branch .hd { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
  .bname { font-weight:600; }
  .badge { font-size:10px; padding:1px 6px; border-radius:9px; border:1px solid; }
  .badge.scripted { color:var(--scripted); border-color:var(--scripted); }
  .badge.rl { color:var(--rl); border-color:var(--rl); }
  .term { color:var(--dim); font-size:12px; padding-left:2px; }
  .term .v { color:var(--fg); }
  .ok { color:var(--applied); } .no { color:var(--opp); }
  /* --- actions --- */
  table.act { width:100%; border-collapse:collapse; font-size:12px; }
  table.act td { padding:2px 4px; }
  table.act td.k { color:var(--dim); width:96px; white-space:nowrap; }
  .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
  /* --- timeline --- */
  .ribbon { position:relative; }
  .ribbon canvas { height:22px; }
  .lbl { color:var(--dim); font-size:11px; width:78px; display:inline-block; }
  input[type=range] { width:100%; accent-color:var(--applied); }
  .legend { color:var(--dim); font-size:11px; display:flex; gap:14px; flex-wrap:wrap; }
  .note { color:var(--dim); font-size:11px; margin-top:8px; line-height:1.45; }
  kbd { background:#21262d; border:1px solid var(--line); border-radius:3px;
        padding:0 4px; font-size:11px; }
</style>
<header>
  <h1>pursuit-evasion · gate explorer</h1>
  <select id="pick"></select>
  <button id="play">▶ play</button>
  <button id="prev">◀</button>
  <button id="next">▶</button>
  <select id="speed">
    <option value="0.5">0.5×</option><option value="1" selected>1×</option>
    <option value="2">2×</option><option value="4">4×</option>
  </select>
  <select id="agent"></select>
  <span id="status" style="color:var(--dim)"></span>
</header>

<div class="grid">
  <div class="stack">
    <div class="panel">
      <h2>top-down (x / y)</h2>
      <canvas id="top" width="760" height="760"></canvas>
      <div class="legend" style="margin-top:8px">
        <span><span class="sw" style="background:var(--scripted)"></span>own team</span>
        <span><span class="sw" style="background:var(--opp)"></span>opponents</span>
        <span><span class="sw" style="background:var(--asset)"></span>asset</span>
        <span><span class="sw" style="background:var(--applied)"></span>applied cmd</span>
        <span id="estLegend" style="display:none"><span class="sw" style="border:1.5px solid var(--est);background:none"></span>believed position (degraded link)</span>
        <span>dashed = the command that controller <em>wanted</em></span>
        <span>dotted ring = capture radius</span>
      </div>
    </div>
    <div class="panel">
      <h2>side (x / z)</h2>
      <canvas id="side" width="1000" height="190"></canvas>
    </div>
  </div>

  <div class="stack">
    <div class="panel">
      <h2 id="treeHd">behaviour tree</h2>
      <div id="tree"></div>
      <div class="note" id="treeNote"></div>
    </div>
    <div class="panel">
      <h2>commands this tick</h2>
      <table class="act" id="acts"></table>
      <div class="note" id="shieldNote"></div>
    </div>
    <div class="panel">
      <h2>features</h2>
      <table class="act" id="feats"></table>
    </div>
  </div>
</div>

<div class="panel" style="margin:0 12px 14px">
  <h2>timeline — who was driving, tick by tick</h2>
  <div id="ribbons"></div>
  <canvas id="dist" width="1400" height="90" style="margin-top:8px"></canvas>
  <input type="range" id="scrub" min="0" value="0">
  <div class="legend" style="margin-top:6px">
    <span><span class="sw" style="background:var(--scripted)"></span>scripted</span>
    <span><span class="sw" style="background:var(--rl)"></span>policy</span>
    <span><span class="sw" style="background:var(--shield)"></span>safety filter corrected</span>
    <span>white line = nearest-opponent distance · gold = asset distance ·
          red dashes = capture radius</span>
    <span><kbd>space</kbd> play <kbd>←</kbd><kbd>→</kbd> step <kbd>a</kbd> agent</span>
  </div>
</div>

<script>
const TRACES = __DATA__;
const C = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let ti = 0, t = 0, agent = 0, playing = false, timer = null;
const D = () => TRACES[ti];
const S = () => D().steps[t];

/* A dead agent contributes no decision, so index by agent id rather than position. */
function decisionFor(step, i) {
  return (step.decisions || []).find(d => d.agent === i) || null;
}
function modeAt(step, i) {
  const d = decisionFor(step, i);
  return d ? d.mode : (step.self_alive && !step.self_alive[i] ? null : "scripted");
}
const num = v => v === null || v === undefined ? "∞"
              : (typeof v === "boolean" ? String(v) : (+v).toFixed(2));
/* inf/nan were nulled for JSON validity; inf is the meaningful one (no solution). */
const val = v => v === null || v === undefined ? Infinity : v;

/* ---------- geometry ---------- */
function fitTop(cv) {
  const R = D().arena.half_extent, m = 18;
  const s = Math.min((cv.width - 2*m) / (2*R), (cv.height - 2*m) / (2*R));
  return { s, px: (x,y) => [cv.width/2 + x*s, cv.height/2 - y*s] };
}
function fitSide(cv) {
  const R = D().arena.half_extent, a = D().arena, m = 16;
  const sx = (cv.width - 2*m) / (2*R);
  const sz = (cv.height - 2*m) / Math.max(a.z_max - a.z_min, 1e-6);
  return { s: sx, px: (x,z) => [cv.width/2 + x*sx, cv.height - m - (z - a.z_min)*sz] };
}

function arrow(g, x, y, dx, dy, color, dash, w) {
  const L = Math.hypot(dx, dy);
  if (L < 1e-6) return;
  g.save(); g.strokeStyle = color; g.fillStyle = color;
  g.lineWidth = w || 2; g.setLineDash(dash || []);
  g.beginPath(); g.moveTo(x, y); g.lineTo(x+dx, y+dy); g.stroke();
  g.setLineDash([]);
  const a = Math.atan2(dy, dx), h = 7;
  g.beginPath(); g.moveTo(x+dx, y+dy);
  g.lineTo(x+dx - h*Math.cos(a-0.4), y+dy - h*Math.sin(a-0.4));
  g.lineTo(x+dx - h*Math.cos(a+0.4), y+dy - h*Math.sin(a+0.4));
  g.closePath(); g.fill(); g.restore();
}

function dot(g, x, y, r, fill, stroke) {
  g.beginPath(); g.arc(x, y, r, 0, 7);
  g.fillStyle = fill; g.fill();
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 2; g.stroke(); }
}

function drawTop() {
  const cv = document.getElementById("top"), g = cv.getContext("2d");
  const d = D(), st = S(), { s, px } = fitTop(cv), R = d.arena.half_extent;
  g.clearRect(0, 0, cv.width, cv.height);

  g.strokeStyle = C("--line"); g.lineWidth = 1;
  g.strokeRect(...px(-R, R), 2*R*s, 2*R*s);
  g.setLineDash([2,6]); g.beginPath();
  g.moveTo(...px(-R,0)); g.lineTo(...px(R,0));
  g.moveTo(...px(0,-R)); g.lineTo(...px(0,R)); g.stroke(); g.setLineDash([]);

  if (d.asset) {
    const [ax, ay] = px(d.asset.pos[0], d.asset.pos[1]);
    g.fillStyle = "rgba(210,153,34,.16)";
    g.beginPath(); g.arc(ax, ay, d.asset.radius*s, 0, 7); g.fill();
    dot(g, ax, ay, 4, C("--asset"));
  }

  /* trails: the last 40 steps, so the geometry of the pass is legible */
  for (const [key, color] of [["self", C("--scripted")], ["opp", C("--opp")]]) {
    for (let i = 0; i < st[key].length; i++) {
      g.beginPath();
      for (let k = Math.max(0, t-40); k <= t; k++) {
        const p = px(d.steps[k][key][i][0], d.steps[k][key][i][1]);
        k === Math.max(0, t-40) ? g.moveTo(...p) : g.lineTo(...p);
      }
      g.strokeStyle = color + "55"; g.lineWidth = 1.5; g.stroke();
    }
  }

  st.opp.forEach((p, i) => {
    const [x, y] = px(p[0], p[1]), alive = st.opp_alive[i];
    dot(g, x, y, 5, alive ? C("--opp") : "#484f58");
    g.fillStyle = C("--dim"); g.font = "10px monospace";
    g.fillText((d.game === "tag" ? "E" : "M") + i + (alive ? "" : " ✕"), x + 8, y - 6);
    if (alive) arrow(g, x, y, st.opp_vel[i][0]*s*0.5, -st.opp_vel[i][1]*s*0.5,
                     C("--opp") + "99", [], 1.5);
  });

  /* truth vs BELIEF: on a degraded link, hollow rings mark where the ground
     station thought each body was, tethered to the truth. A label swap shows
     as two long crossed tethers — the estimate is spatially perfect and still
     wrong, which is the whole point of rendering it. */
  if (st.est) {
    st.est.opp.forEach((p, i) => {
      const [ex, ey] = px(p[0], p[1]);
      const [tx, ty] = px(st.opp[i][0], st.opp[i][1]);
      if (Math.hypot(ex-tx, ey-ty) > 2) {
        g.setLineDash([3,3]); g.strokeStyle = C("--est"); g.lineWidth = 1;
        g.beginPath(); g.moveTo(tx, ty); g.lineTo(ex, ey); g.stroke(); g.setLineDash([]);
      }
      const swapped = st.est.swapped && st.est.swapped.includes(i);
      g.strokeStyle = swapped ? "#f85149" : C("--est"); g.lineWidth = swapped ? 2 : 1.2;
      g.beginPath(); g.arc(ex, ey, 6, 0, 7); g.stroke();
      if (swapped) { g.fillStyle = "#f85149"; g.font = "9px monospace";
                     g.fillText("swap", ex + 8, ey + 10); }
    });
    st.est.self.forEach((p, i) => {
      const [ex, ey] = px(p[0], p[1]);
      g.strokeStyle = C("--est"); g.lineWidth = 1.2;
      g.beginPath(); g.arc(ex, ey, 6, 0, 7); g.stroke();
    });
  }

  st.self.forEach((p, i) => {
    const [x, y] = px(p[0], p[1]), sel = i === agent;
    g.setLineDash([2,3]); g.strokeStyle = "#ffffff28"; g.lineWidth = 1;
    g.beginPath(); g.arc(x, y, d.capture_radius*s, 0, 7); g.stroke(); g.setLineDash([]);
    const m = modeAt(st, i);
    dot(g, x, y, sel ? 7 : 5, m === "rl" ? C("--rl") : C("--scripted"),
        sel ? "#fff" : null);
    g.fillStyle = C("--dim"); g.font = "10px monospace";
    g.fillText((d.game === "tag" ? "P" : "D") + i, x + 9, y - 7);
  });

  /* command arrows for the selected agent — unit commands, so scale generously */
  const st_ = st, dec = decisionFor(st_, agent);
  if (st_.self[agent]) {
    const [x, y] = px(st_.self[agent][0], st_.self[agent][1]), K = 6*s;
    if (dec) {
      arrow(g, x, y, dec.scripted[0]*K, -dec.scripted[1]*K, C("--scripted"), [5,4]);
      arrow(g, x, y, dec.rl[0]*K, -dec.rl[1]*K, C("--rl"), [5,4]);
    }
    const a = st_.applied[agent];
    arrow(g, x, y, a[0]*K, -a[1]*K, C("--applied"), [], 2.5);
  }
}

function drawSide() {
  const cv = document.getElementById("side"), g = cv.getContext("2d");
  const d = D(), st = S(), { s, px } = fitSide(cv), R = d.arena.half_extent;
  g.clearRect(0, 0, cv.width, cv.height);
  g.strokeStyle = C("--line"); g.lineWidth = 1;
  g.beginPath();
  g.moveTo(...px(-R, d.arena.z_min)); g.lineTo(...px(R, d.arena.z_min));
  g.moveTo(...px(-R, d.arena.z_max)); g.lineTo(...px(R, d.arena.z_max));
  g.stroke();
  g.fillStyle = C("--dim"); g.font = "10px monospace";
  g.fillText("z=" + d.arena.z_max, 4, 12);
  g.fillText("z=" + d.arena.z_min, 4, cv.height - 6);
  if (d.asset) dot(g, ...px(d.asset.pos[0], d.asset.pos[2]), 4, C("--asset"));
  st.opp.forEach((p, i) =>
    dot(g, ...px(p[0], p[2]), 5, st.opp_alive[i] ? C("--opp") : "#484f58"));
  st.self.forEach((p, i) =>
    dot(g, ...px(p[0], p[2]), i === agent ? 7 : 5,
        modeAt(st, i) === "rl" ? C("--rl") : C("--scripted"), i === agent ? "#fff" : null));
}

/* ---------- behaviour tree panel ---------- */
function termHtml(term, feats, thr) {
  const [f, op, ref] = term;
  const tv = (typeof ref === "string") ? thr[ref] : ref;
  const v = feats[f];
  let pass;
  if (op === "is") pass = (!!v) === (!!tv);
  else if (op === "<") pass = val(v) < tv;
  else if (op === "<=") pass = val(v) <= tv;
  else if (op === ">") pass = val(v) > tv;
  else pass = val(v) >= tv;
  /* written as the comparison, with the live value in brackets — so you can see
     not just whether it passed but by how much it missed */
  const rhs = (typeof ref === "string") ? `${ref}=${num(tv)}` : num(tv);
  return `<div class="term">${pass ? '<span class="ok">✓</span>' : '<span class="no">✗</span>'}
    ${f} ${op === "is" ? "==" : op} ${rhs} <span class="v">[${num(v)}]</span></div>`;
}

function drawTree() {
  const d = D(), st = S(), dec = decisionFor(st, agent);
  const hd = document.getElementById("treeHd"), box = document.getElementById("tree");
  const note = document.getElementById("treeNote");
  if (!dec) {
    hd.textContent = "behaviour tree";
    box.innerHTML = `<div class="term">no gate on this run — the
      <b>${d.controller}</b> controller drives every agent directly.</div>`;
    note.textContent = "";
    return;
  }
  hd.textContent = `selector gate[${d.profile}]`;
  const fired = d.branches.findIndex(b => b.name === dec.branch);
  box.innerHTML = d.branches.map((b, i) => {
    const cls = i === fired ? "fired" : (i > fired ? "unreached" : "");
    const terms = b.reads.map(r => termHtml(r, dec.features, d.thresholds)).join("");
    return `<div class="branch ${cls}">
      <div class="hd"><span class="bname">${i === fired ? "▶ " : ""}${b.name}</span>
        <span class="badge ${b.mode}">${b.mode === "rl" ? "POLICY" : "SCRIPTED"}</span></div>
      ${terms || '<div class="term">fallback — always succeeds</div>'}</div>`;
  }).join("");
  note.innerHTML = `A py_trees Selector returns on the first child that succeeds, so
    the greyed branches below <b>${dec.branch}</b> were never ticked this step.`;
}

function drawActs() {
  const d = D(), st = S(), dec = decisionFor(st, agent);
  const fmt = v => v.map(c => (c >= 0 ? " " : "") + c.toFixed(2)).join("  ");
  let rows = "";
  if (dec) {
    const chosen = dec.mode === "rl" ? "rl" : "scripted";
    rows += `<tr><td class="k"><span class="sw" style="background:var(--scripted)"></span>scripted</td>
             <td>${fmt(dec.scripted)}${chosen === "scripted" ? "  ← gate" : ""}</td></tr>`;
    rows += `<tr><td class="k"><span class="sw" style="background:var(--rl)"></span>policy</td>
             <td>${fmt(dec.rl)}${chosen === "rl" ? "  ← gate" : ""}</td></tr>`;
  }
  rows += `<tr><td class="k"><span class="sw" style="background:var(--applied)"></span>applied</td>
           <td>${fmt(st.applied[agent])}</td></tr>`;
  document.getElementById("acts").innerHTML = rows;

  const sh = st.shield || { geofence: 0, speed: 0 };
  const n = sh.geofence + sh.speed;
  const el = document.getElementById("shieldNote");
  if (n > 0) {
    el.innerHTML = `<span style="color:var(--shield)">safety filter corrected this tick</span>
      — geofence ${sh.geofence}, speed ${sh.speed}. The shield sits <em>below</em> both
      the tree and the policy, so "applied" is what the vehicle got regardless of
      which controller the gate chose.`;
  } else {
    el.innerHTML = `safety filter passed this tick unchanged.`;
  }
}

function drawFeats() {
  const dec = decisionFor(S(), agent);
  if (!dec) {
    document.getElementById("feats").innerHTML =
      `<tr><td class="k" colspan="2">features are only recorded where a gate
        evaluated them — this run has none.</td></tr>`;
    return;
  }
  document.getElementById("feats").innerHTML = Object.entries(dec.features)
    .map(([k, v]) => `<tr><td class="k" style="width:130px">${k}</td>
                      <td>${num(v)}</td></tr>`).join("");
}

/* ---------- timeline ---------- */
function buildRibbons() {
  const d = D(), n = d.steps[0].self.length;
  const wrap = document.getElementById("ribbons");
  wrap.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="lbl">${(d.game === "tag" ? "pursuer " : "defender ")}${i}</span>`;
    const cv = document.createElement("canvas");
    cv.width = 1400; cv.height = 22; cv.style.flex = "1";
    cv.dataset.agent = i;
    cv.onclick = e => { const r = cv.getBoundingClientRect();
      setStep(Math.round((e.clientX - r.left) / r.width * (d.steps.length - 1)));
      setAgent(i); };
    row.appendChild(cv); wrap.appendChild(row);
  }
}

function drawRibbons() {
  const d = D(), N = d.steps.length;
  document.querySelectorAll("#ribbons canvas").forEach(cv => {
    const i = +cv.dataset.agent, g = cv.getContext("2d");
    const w = cv.width / N;
    g.clearRect(0, 0, cv.width, cv.height);
    d.steps.forEach((st, k) => {
      const m = modeAt(st, i);
      g.fillStyle = m === null ? "#30363d" : (m === "rl" ? C("--rl") : C("--scripted"));
      g.fillRect(k*w, 3, w + 1, 12);          // +1: no hairline seams between ticks
      const sh = st.shield || {};
      if ((sh.geofence || 0) + (sh.speed || 0) > 0) {
        g.fillStyle = C("--shield"); g.fillRect(k*w, 16, w + 1, 4);
      }
    });
    /* a cursor, not a block: short episodes make w tens of pixels wide */
    g.fillStyle = "#fff"; g.fillRect(t*w + w/2 - 1, 0, 2, cv.height);
    if (i === agent) { g.strokeStyle = "#ffffff44"; g.strokeRect(0.5, 0.5, cv.width-1, cv.height-1); }
  });
}

function drawDist() {
  const cv = document.getElementById("dist"), g = cv.getContext("2d"), d = D();
  const N = d.steps.length, H = cv.height, pad = 6;
  g.clearRect(0, 0, cv.width, H);
  const vals = d.steps.flatMap(s => [val(s.min_dist), s.asset_dist === null ? 0 : val(s.asset_dist)])
                      .filter(v => isFinite(v));
  const max = Math.max(...vals, d.capture_radius * 2, 1);
  const Y = v => H - pad - (Math.min(val(v), max) / max) * (H - 2*pad);
  const X = k => k / Math.max(N - 1, 1) * cv.width;

  g.strokeStyle = C("--opp"); g.setLineDash([4,4]); g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, Y(d.capture_radius)); g.lineTo(cv.width, Y(d.capture_radius));
  g.stroke(); g.setLineDash([]);

  for (const [key, color] of [["min_dist", "#e6edf3"], ["asset_dist", C("--asset")]]) {
    if (d.steps[0][key] === null && key === "asset_dist") continue;
    g.beginPath();
    d.steps.forEach((s, k) => k ? g.lineTo(X(k), Y(s[key])) : g.moveTo(X(k), Y(s[key])));
    g.strokeStyle = color; g.lineWidth = 1.5; g.stroke();
  }
  g.fillStyle = "#fff"; g.fillRect(X(t), 0, 1.5, H);
  g.fillStyle = C("--dim"); g.font = "10px monospace";
  g.fillText(max.toFixed(0) + " m", 4, 16);
  g.fillText("capture " + d.capture_radius + " m", 4, Y(d.capture_radius) - 4);
}

/* ---------- wiring ---------- */
function setStep(k) {
  t = Math.max(0, Math.min(D().steps.length - 1, k));
  document.getElementById("scrub").value = t;
  render();
}
function setAgent(i) { agent = i; document.getElementById("agent").value = i; render(); }

function render() {
  const d = D(), o = d.outcome;
  const win = d.game === "tag"
    ? (o.pursuer_win ? "pursuers win" : "evaders survive")
    : (o.breach ? "BREACH — attackers win" : "asset held — defenders win");
  const share = Object.values(d.mode_counts).reduce((a, b) => a + b, 0);
  const lk = D().link && D().link !== "perfect"
    ? ` · link ${D().link} (${(D().link_stats||{}).swaps||0} swaps, ${(D().link_stats||{}).dropouts||0} drops)` : "";
  document.getElementById("status").textContent =
    `step ${t+1}/${d.steps.length}  ·  t=${(t*d.dt).toFixed(2)}s  ·  ${win}` +
    (share ? `  ·  policy drove ${d.mode_counts.rl}/${share} agent-ticks` : "") + lk;
  /* keep the URL pointing at exactly this moment, so a finding is linkable */
  history.replaceState(null, "", `#${ti}/${t}/${agent}`);
  drawTop(); drawSide(); drawTree(); drawActs(); drawFeats();
  drawRibbons(); drawDist();
}

function loadTrace(k) {
  ti = k; t = 0; agent = 0;
  const d = D();
  document.getElementById("scrub").max = d.steps.length - 1;
  document.getElementById("scrub").value = 0;
  const ag = document.getElementById("agent");
  ag.innerHTML = d.steps[0].self
    .map((_, i) => `<option value="${i}">${d.game === "tag" ? "pursuer" : "defender"} ${i}</option>`)
    .join("");
  document.getElementById("estLegend").style.display =
    (d.link && d.link !== "perfect") ? "" : "none";
  buildRibbons();
  render();
}

function play(on) {
  playing = on === undefined ? !playing : on;
  document.getElementById("play").textContent = playing ? "⏸ pause" : "▶ play";
  document.getElementById("play").classList.toggle("on", playing);
  clearInterval(timer);
  if (playing) {
    const ms = D().dt * 1000 / +document.getElementById("speed").value;
    timer = setInterval(() => {
      if (t >= D().steps.length - 1) { play(false); return; }
      setStep(t + 1);
    }, ms);
  }
}

document.getElementById("pick").innerHTML = TRACES
  .map((d, i) => `<option value="${i}">${d.game} · ${d.controller} · seed ${d.seed}${d.link && d.link !== "perfect" ? " · " + d.link : ""}</option>`)
  .join("");
document.getElementById("pick").onchange = e => { play(false); loadTrace(+e.target.value); };
document.getElementById("agent").onchange = e => setAgent(+e.target.value);
document.getElementById("scrub").oninput = e => setStep(+e.target.value);
document.getElementById("play").onclick = () => play();
document.getElementById("prev").onclick = () => { play(false); setStep(t - 1); };
document.getElementById("next").onclick = () => { play(false); setStep(t + 1); };
document.getElementById("speed").onchange = () => { if (playing) play(true); };
addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "ArrowLeft") { play(false); setStep(t - 1); }
  else if (e.key === "ArrowRight") { play(false); setStep(t + 1); }
  else if (e.key === "a") setAgent((agent + 1) % D().steps[0].self.length);
});
/* #trace/step/agent — deep link straight to a moment worth arguing about */
const H = (location.hash || "").slice(1).split("/").map(Number);
loadTrace(Number.isFinite(H[0]) && TRACES[H[0]] ? H[0] : 0);
document.getElementById("pick").value = ti;
if (Number.isFinite(H[2])) setAgent(H[2]);
if (Number.isFinite(H[1])) setStep(H[1]);
</script>
"""


def build(traces: list[dict], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_TEMPLATE.replace("__DATA__", json.dumps(traces, separators=(",", ":"))))
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", default="assault", choices=list(GAME_KEYS))
    p.add_argument("--controllers", nargs="+", default=["scripted", "bt", "bt_safe"],
                   choices=["scripted", "rl", "bt", "bt_safe"],
                   help="recorded on the same seed, so the dropdown is an A/B")
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--link", nargs="+", default=["perfect"],
                   help="record each controller under each of these link presets "
                        "(same seed, same corruption) — e.g. --link perfect "
                        "vicon_busy puts truth-vs-belief side by side")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    model = args.model or ("models/pursuer_dagger.zip" if args.game == "tag"
                           else f"models/{args.game}_dagger.zip")
    traces = []
    for kind in args.controllers:
        for link in args.link:
            tr = record(args.game, kind, model, args.seed, link=link)
            tag = f"{kind}" + ("" if link == "perfect" else f"@{link}")
            print(f"  [{tag}] {len(tr['steps'])} steps, modes={tr['mode_counts']}, "
                  f"outcome={tr['outcome']}, link={tr['link_stats']}")
            traces.append(tr)

    suffix = "" if args.link == ["perfect"] else "_" + "_".join(
        l for l in args.link if l != "perfect")
    dest = args.out or Path(f"results/explorer_{args.game}_{args.seed}{suffix}.html")
    build(traces, dest)
    kb = dest.stat().st_size / 1024
    print(f"\n[explorer] {len(traces)} traces -> {dest} ({kb:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
