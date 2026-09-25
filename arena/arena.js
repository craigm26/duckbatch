// The walker arena: every valid Microduck walking policy, one field, one set of controls.
//
// WHY ONE CONTROL FOR ALL OF THEM. A teacher and its distilled students differ by a few
// percent in tracking and by a factor in falls (craigm26/duckbatch b002, p001). Watching one
// duck at a time, that is invisible; driving them together, it is the thing you see: the
// same stick, the same push, the same instant, and one duck wobbles or goes down.
//
// HOW IT IS FAIR. Every policy gets its OWN physics world — its own MjData on one shared
// MjModel, reset to the same state — so ducks cannot collide, and the only difference between
// lanes is the network. Each world is stepped at 0.005 s, and its policy runs every 4th step
// (50 Hz), exactly as craigm26/duckbench's walk demo does it (control = HOME + action).
//
// WHAT IT IS NOT. This scene is plain MuJoCo (duckbench's scene.mjb), not mjlab's training
// environment with its actuator model, so a lane here is also a small sim-to-sim transfer test.
// It says which walker looks steadier under the same inputs; the measured comparison is
// duckbatch's (notes/2026-09-24-b002-close.md, p001-close.md).
import loadMuJoCo from './vendor/mujoco.js';
import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { makeLoop } from './duckloop.mjs';
import { findStairJoints, clearStairs } from './stairs.js';
import { createArenaRenderer } from './arena-render.js';

ort.env.wasm.wasmPaths = new URL('./vendor/ort/', document.baseURI).href;
ort.env.wasm.numThreads = 1;

const DECIMATION = 4;
const TIMESTEP = 0.005;              // duckbench's scene: 0.005 s physics, 50 Hz policy
const LANE_GAP = 0.45;               // metres between lanes: a duck's stance is ~0.15 m wide
const COLOURS = [[0.16, 0.47, 0.84], [0.92, 0.41, 0.20], [0.11, 0.69, 0.48], [0.91, 0.63, 0.0],
                 [0.91, 0.48, 0.64], [0.29, 0.23, 0.65], [0.0, 0.51, 0.0], [0.89, 0.29, 0.28]];
const HF = 'https://huggingface.co';

const $ = s => document.querySelector(s);
const status = $('#status');
let mj, model, C, loop, renderer, GYRO = 0, DUCK, STAIRS;
const lanes = [];
const cmd = { vx: 0, vy: 0, vyaw: 0 };
let running = true, simTime = 0, orbit = 0.35, zoom = 1;

// ---------------------------------------------------------------- policies

function resolveRef(ref) {
  // "org/repo" -> the repo's policy.onnx + manifest.json; "org/repo@rev:file.onnx" -> that file.
  const m = /^([\w.-]+\/[\w.-]+)(?:@([\w.-]+))?(?::([\w./-]+\.onnx))?$/.exec(ref.trim());
  if (!m) throw new Error(`"${ref}" is not org/repo[@revision][:file.onnx]`);
  const [, repo, rev = 'main', file = 'policy.onnx'] = m;
  return { repo, rev, file, onnx: `${HF}/${repo}/resolve/${rev}/${file}`,
           manifest: file === 'policy.onnx' ? `${HF}/${repo}/resolve/${rev}/manifest.json` : null };
}

/** The same checks Pollen's simulator makes before a community move touches its duck:
 *  one input, one output, [1,14] finite actions, and not a constant network. */
async function validate(session) {
  if (session.inputNames.length !== 1 || session.outputNames.length !== 1) {
    throw new Error(`expected 1 input / 1 output, got ${session.inputNames.length}/${session.outputNames.length}`);
  }
  const run = async fill => {
    const buf = new Float32Array(61);
    if (fill) for (let i = 0; i < 61; i++) buf[i] = (Math.random() - 0.5) * 0.1;
    const out = (await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', buf, [1, 61]) }))[session.outputNames[0]];
    if (out.dims.length !== 2 || out.dims[1] !== 14) throw new Error(`output is [${out.dims}], not [1,14]`);
    if (!Array.from(out.data).every(Number.isFinite)) throw new Error('non-finite actions');
    return Array.from(out.data);
  };
  await run(false);
  const a = await run(true), b = await run(true);
  if (a.every((v, i) => Math.abs(v - b[i]) < 1e-9)) throw new Error('constant output: a dead network');
}

async function addPolicy(entry) {
  const ref = resolveRef(entry.ref);
  const label = entry.label || ref.repo.split('/')[1];
  const row = addRow(label, 'loading…');
  try {
    const bytes = new Uint8Array(await (await fetchOk(ref.onnx, 'policy')).arrayBuffer());
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    await validate(session);
    let manifest = null;
    if (ref.manifest) { try { manifest = await (await fetchOk(ref.manifest, 'manifest')).json(); } catch { /* optional */ } }
    if (manifest && (manifest.obs_len ?? 61) !== 61) throw new Error(`manifest says obs_len ${manifest.obs_len}`);
    if (manifest && manifest.kind === 'episodic') throw new Error('a one-shot move, not a walker');
    const lane = {
      ref: entry.ref, label, note: entry.note || manifest?.description || '', role: entry.role || '',
      session, input: session.inputNames[0], output: session.outputNames[0],
      data: new mj.MjData(model), colour: COLOURS[lanes.length % COLOURS.length],
      visible: true, row, bytes: bytes.length,
    };
    lanes.push(lane);
    layoutLanes();
    resetLane(lane);
    paintRow(lane);
  } catch (e) {
    row.querySelector('.stats').textContent = `refused: ${e.message}`;
    row.classList.add('refused');
  }
}

async function fetchOk(url, what) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status === 401 || r.status === 404 ? `no ${what} at ${url}` : `${what}: HTTP ${r.status}`);
  return r;
}

// ---------------------------------------------------------------- physics

function layoutLanes() {
  const n = lanes.length;
  lanes.forEach((l, i) => { l.laneY = (i - (n - 1) / 2) * LANE_GAP; });
}

function resetLane(l) {
  const d = l.data;
  mj.mj_resetData(model, d);
  d.qpos[DUCK.freeQpos + 2] = 0.12; d.qpos[DUCK.freeQpos + 3] = 1;
  for (let i = 0; i < 14; i++) { d.qpos[DUCK.qpos[i]] = loop.HOME[i]; d.ctrl[i] = loop.HOME[i]; }
  if (STAIRS) clearStairs(d, STAIRS);
  mj.mj_forward(model, d);
  l.lastAction = new Array(14).fill(0);
  l.prevAction = null;
  l.stats = { ticks: 0, upTicks: 0, falls: 0, down: false, linErr: 0, angErr: 0, jitter: 0,
              startX: d.qpos[DUCK.freeQpos], startY: d.qpos[DUCK.freeQpos + 1] };
}

function resetAll() { simTime = 0; autoPushed = 0; for (const l of lanes) resetLane(l); }

/** The same push, in the same direction, to every duck at once. */
function pushAll() {
  const ang = Math.random() * 2 * Math.PI, v = 0.35;
  for (const l of lanes) {
    l.data.qvel[DUCK.freeDof] += v * Math.cos(ang);
    l.data.qvel[DUCK.freeDof + 1] += v * Math.sin(ang);
  }
  flash(`pushed every duck ${v} m/s towards ${Math.round(ang * 180 / Math.PI)}°`);
}

async function tickAll() {
  const command = loop.command(cmd);
  await Promise.all(lanes.map(async l => {
    const d = l.data, f = DUCK.freeQpos;
    const q = [d.qpos[f + 3], d.qpos[f + 4], d.qpos[f + 5], d.qpos[f + 6]];
    const grav = loop.projectedGravity(q);
    const gyro = [d.sensordata[GYRO], d.sensordata[GYRO + 1], d.sensordata[GYRO + 2]];
    const jpos = [], jvel = [];
    for (let k = 0; k < 14; k++) { jpos.push(d.qpos[DUCK.qpos[k]]); jvel.push(d.qvel[DUCK.dof[k]]); }
    const obs = loop.buildObs(gyro, grav, jpos, jvel, l.lastAction, command);
    const out = await l.session.run({ [l.input]: new ort.Tensor('float32', obs, [1, 61]) });
    const a = Array.from(out[l.output].data);
    for (let k = 0; k < 14; k++) d.ctrl[k] = loop.HOME[k] + a[k];
    // Live measurements, the same definitions as duckbatch's walk eval.
    const s = l.stats, down = grav[2] > -0.5;
    if (down && !s.down) s.falls++;
    s.down = down;
    {
      const vx0 = d.qvel[DUCK.freeDof], vy0 = d.qvel[DUCK.freeDof + 1];
      s.speedNow = 0.9 * (s.speedNow || 0) + 0.1 * Math.hypot(vx0, vy0);
    }
    if (!down) {
      s.upTicks++;
      const vx = d.qvel[DUCK.freeDof], vy = d.qvel[DUCK.freeDof + 1], wz = d.qvel[DUCK.freeDof + 5];
      // Body-frame planar speed from the world velocity and the trunk's yaw.
      const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
      const bx = Math.cos(yaw) * vx + Math.sin(yaw) * vy, by = -Math.sin(yaw) * vx + Math.cos(yaw) * vy;
      s.linErr += Math.hypot(cmd.vx - bx, cmd.vy - by);
      s.angErr += Math.abs(cmd.vyaw - wz);
    }
    if (l.prevAction) { let j = 0; for (let k = 0; k < 14; k++) j += Math.abs(a[k] - l.prevAction[k]); s.jitter += j / 14; }
    l.prevAction = a; l.lastAction = a; s.ticks++;
  }));
  for (const l of lanes) {
    if (STAIRS) clearStairs(l.data, STAIRS);
    for (let s = 0; s < DECIMATION; s++) mj.mj_step(model, l.data);
  }
  simTime += DECIMATION * TIMESTEP;
}

// ---------------------------------------------------------------- controls

const keys = new Set();
addEventListener('keydown', e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  keys.add(e.key.toLowerCase());
  if (e.key === ' ') { e.preventDefault(); running = !running; }
  if (e.key.toLowerCase() === 'r') resetAll();
  if (e.key.toLowerCase() === 'p') pushAll();
});
addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
// ON-SCREEN BUTTONS LATCH; KEYS ARE HELD. A click or a tap is a tenth of a second, and a
// tenth of a second of "forward" moves no duck — so a tap sets the direction and it stays
// set until the same button is tapped again or Stop is. Keys and the stick stay hold-to-drive.
const latch = { x: 0, t: 0 };
function paintLatch() {
  for (const b of document.querySelectorAll('[data-latch]')) {
    const k = b.dataset.latch;
    const on = (k === 'up' && latch.x > 0) || (k === 'down' && latch.x < 0) ||
               (k === 'left' && latch.t > 0) || (k === 'right' && latch.t < 0);
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
}
for (const b of document.querySelectorAll('[data-latch]')) {
  b.addEventListener('click', () => {
    const k = b.dataset.latch;
    if (k === 'stop') { latch.x = 0; latch.t = 0; }
    if (k === 'up') latch.x = latch.x > 0 ? 0 : 1;
    if (k === 'down') latch.x = latch.x < 0 ? 0 : -1;
    if (k === 'left') latch.t = latch.t > 0 ? 0 : 1;
    if (k === 'right') latch.t = latch.t < 0 ? 0 : -1;
    paintLatch();
  });
}

const padWas = { r: false, p: false };
const AUTO = new URLSearchParams(location.search).get('auto') === '1';
let autoPushed = 0;
function readControls() {
  const fwd = (keys.has('arrowup') || keys.has('w')) ? 1 : 0;
  const back = (keys.has('arrowdown') || keys.has('s')) ? 1 : 0;
  const left = (keys.has('arrowleft') || keys.has('a')) ? 1 : 0;
  const right = (keys.has('arrowright') || keys.has('d')) ? 1 : 0;
  let x = fwd - back || latch.x, t = left - right || latch.t;
  const pad = navigator.getGamepads ? [...navigator.getGamepads()].find(Boolean) : null;
  if (pad) {
    const dz = v => (Math.abs(v) < 0.15 ? 0 : v);
    if (dz(pad.axes[1]) || dz(pad.axes[0])) { x = -dz(pad.axes[1]); t = -dz(pad.axes[0]); }
    // Edges, not levels: the browser hands back a fresh pad object every poll.
    const r = !!pad.buttons[2]?.pressed, p = !!pad.buttons[1]?.pressed;
    if (r && !padWas.r) resetAll();
    if (p && !padWas.p) pushAll();
    padWas.r = r; padWas.p = p;
  }
  if (AUTO && !x && !t) {
    // Autopilot (?auto=1): walk forward, and push everyone every 6 s of sim time.
    x = 0.8;
    if (simTime - autoPushed >= 6) { autoPushed = simTime; pushAll(); }
  }
  const scale = +$('#speed').value;
  // THE TRAINING RANGE, NOT POLLEN'S KEYBOARD LIMIT. VelStand was trained on vx in ±0.4 and
  // wz in ±1.0, and it has a dead band: in duckbench's scene velstand stands still below about
  // 0.3 m/s and the 128-128 student below about 0.4 (measured, 8 s runs). Pollen's 0.25 cap
  // suits alpha_walking and leaves every VelStand walker standing, which looked like a bug.
  cmd.vx = x > 0 ? 0.4 * x * scale : 0.3 * x * scale;
  cmd.vyaw = 1.0 * t * scale;
  $('#cmd').textContent = `command  vx ${cmd.vx.toFixed(2)} m/s   yaw ${cmd.vyaw.toFixed(2)} rad/s`;
}

// ---------------------------------------------------------------- panel

function addRow(label, text) {
  const row = document.createElement('div');
  row.className = 'lane';
  row.innerHTML = `<label><input type="checkbox" checked><span class="sw"></span><b></b></label>
                   <div class="stats"></div><div class="note"></div>`;
  row.querySelector('b').textContent = label;
  row.querySelector('.stats').textContent = text;
  $('#lanes').appendChild(row);
  return row;
}

function paintRow(l) {
  const [r, g, b] = l.colour.map(v => Math.round(v * 255));
  l.row.querySelector('.sw').style.background = `rgb(${r},${g},${b})`;
  l.row.querySelector('.note').textContent = [l.role, l.note].filter(Boolean).join(' · ');
  l.row.querySelector('input').onchange = e => { l.visible = e.target.checked; };
}

function paintStats() {
  for (const l of lanes) {
    const s = l.stats, up = Math.max(s.upTicks, 1);
    // Planar displacement, not x alone: headings drift, and a duck that curved is not a slow one.
    const dist = Math.hypot(l.data.qpos[DUCK.freeQpos] - s.startX, l.data.qpos[DUCK.freeQpos + 1] - s.startY);
    l.row.querySelector('.stats').textContent =
      `now ${(s.speedNow || 0).toFixed(2)} m/s · falls ${s.falls}${s.down ? ' (down)' : ''} · track err ${(s.linErr / up).toFixed(3)} m/s, ` +
      `${(s.angErr / up).toFixed(2)} rad/s · jitter ${(s.jitter / Math.max(s.ticks - 1, 1)).toFixed(4)} · ` +
      `${dist.toFixed(2)} m`;
  }
  $('#clock').textContent = `sim ${simTime.toFixed(1)} s`;
}

let flashTimer = 0;
function flash(text) { status.textContent = text; clearTimeout(flashTimer); flashTimer = setTimeout(() => { status.textContent = ''; }, 2500); }

// ---------------------------------------------------------------- frame loop

let last = performance.now(), acc = 0, busy = false;
async function frame(now) {
  requestAnimationFrame(frame);
  readControls();
  if (running && !busy && lanes.length) {
    acc += Math.min((now - last) / 1000, 0.1);
    busy = true;
    // Real time where the device keeps up; slower (never skipping ticks) where it does not.
    let steps = 0;
    const dt = DECIMATION * TIMESTEP;
    while (acc >= dt && steps < 4) { await tickAll(); acc -= dt; steps++; }
    if (steps === 4) acc = 0;
    busy = false;
  }
  last = now;
  if (lanes.length) {
    const shown = lanes.filter(l => l.visible);
    const focusX = shown.length ? shown.reduce((s, l) => s + l.data.qpos[DUCK.freeQpos], 0) / shown.length : 0;
    renderer.render(lanes.map(l => ({ data: l.data, laneY: l.laneY, colour: l.colour, root: DUCK.freeQpos, visible: l.visible })),
                    { model, bg: [0.96, 0.96, 0.95], grid: [0.82, 0.82, 0.8], zoom, focusX, orbit });
    paintStats();
  }
}

// ---------------------------------------------------------------- start

(async function start() {
  const params = new URLSearchParams(location.search);
  try {
    status.textContent = 'loading physics…';
    C = await (await fetch('./duckkit-constants.json')).json();
    loop = makeLoop(C);
    mj = await loadMuJoCo();
    const mjb = await (await fetch('./scene.mjb')).arrayBuffer();
    mj.FS.writeFile('/scene.mjb', new Uint8Array(mjb));
    model = mj.MjModel.mj_loadBinary('/scene.mjb', new mj.MjVFS());
    for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
    DUCK = loop.findDuckJoints(model);
    STAIRS = findStairJoints(model);
    renderer = await createArenaRenderer($('#view'), './duck-visual.bin');
    status.textContent = 'loading walkers…';
    const extra = (params.get('move') || '').split(',').filter(Boolean).map(ref => ({ ref, role: 'added by link' }));
    const listed = params.get('only') ? [] : (await (await fetch('./policies.json')).json()).policies;
    for (const entry of [...listed, ...extra]) await addPolicy(entry);
    status.textContent = '';
    window.__arena = { lanes, get simTime() { return simTime; } };
    console.log('[arena] ready: ' + lanes.map(l => l.label).join(', '));
    const selftest = +(params.get('selftest') || 0);
    if (selftest) {
      // ?selftest=N: N control ticks walking forward, with one shared push halfway,
      // then every lane's numbers on the page. For headless checks; no animation.
      // ?press=up[,left]: click those on-screen buttons first, then drive through the same
      // readControls() the live page uses — the path a person tapping actually takes.
      for (const k of (params.get('press') || 'up').split(',')) document.querySelector(`[data-latch="${k}"]`)?.click();
      const t0 = performance.now();
      for (let i = 0; i < selftest; i++) {
        readControls();
        if (i === Math.floor(selftest / 2)) pushAll();
        await tickAll();
      }
      const ms = (performance.now() - t0) / selftest;
      paintStats();
      const out = lanes.map(l => `${l.label}: ${l.row.querySelector('.stats').textContent}`).join('\n');
      status.textContent = `selftest ${selftest} ticks, ${ms.toFixed(2)} ms per tick for ${lanes.length} lanes`;
      console.log('[arena] selftest ' + ms.toFixed(2) + ' ms/tick\n' + out);
      renderer.render(lanes.map(l => ({ data: l.data, laneY: l.laneY, colour: l.colour, root: DUCK.freeQpos, visible: true })),
                      { model, bg: [0.96, 0.96, 0.95], grid: [0.82, 0.82, 0.8], zoom, focusX: lanes.reduce((s, l) => s + l.data.qpos[DUCK.freeQpos], 0) / lanes.length, orbit });
      return;
    }
    requestAnimationFrame(frame);
  } catch (e) {
    status.textContent = 'could not start: ' + e.message;
    throw e;
  }
})();

$('#reset').onclick = resetAll;
$('#push').onclick = pushAll;
$('#pause').onclick = () => { running = !running; };
$('#zoom').oninput = e => { zoom = +e.target.value; };
$('#orbit').oninput = e => { orbit = +e.target.value; };
$('#add').onsubmit = async e => {
  e.preventDefault();
  const ref = $('#addRef').value.trim();
  if (!ref) return;
  await addPolicy({ ref, role: 'added here' });
  $('#addRef').value = '';
};
$('#discover').onclick = async () => {
  // Community walkers: Hub repos tagged microduck-policy whose manifest is a walk-slot gait.
  const btn = $('#discover'); btn.disabled = true; btn.textContent = 'searching…';
  try {
    const repos = await (await fetch(`${HF}/api/models?filter=microduck-policy&limit=50`)).json();
    const have = new Set(lanes.map(l => resolveRef(l.ref).repo));
    let added = 0;
    for (const r of repos) {
      if (have.has(r.id)) continue;
      let m = null;
      try { m = await (await fetchOk(`${HF}/${r.id}/resolve/main/manifest.json`, 'manifest')).json(); } catch { continue; }
      if ((m.kind ?? 'perpetual') !== 'perpetual' || (m.slot && m.slot !== 'walk') || m.mode === 'roller') continue;
      await addPolicy({ ref: r.id, role: 'community' }); added++;
    }
    btn.textContent = added ? `added ${added}` : 'no others found';
  } catch (e) { btn.textContent = 'search failed'; flash(e.message); }
};
