// Many ducks, one camera: the arena's renderer.
//
// A FORK OF craigm26/duckbench site/render.js, NOT A NEW RENDERER. The mesh
// pack (duck-visual.bin, exported from MuJoCo's own compiled model), the
// shaders and the lighting are that file's, so a duck here looks exactly like
// the duck in the walk demo. What changes is the scene: every policy gets its
// OWN physics world (its own MjData on one shared MjModel, so ducks cannot
// bump into each other), and each world is drawn in its own lane, shifted
// sideways, under one camera that follows the whole field.

function mat4() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; }
function multiply(out, a, b) {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return out;
}
/** MuJoCo quaternions are [w, x, y, z]. */
function fromQuatPos(out, q, p) {
  const [w, x, y, z] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy; out[3] = 0;
  out[4] = xy - wz; out[5] = 1 - (xx + zz); out[6] = yz + wx; out[7] = 0;
  out[8] = xz + wy; out[9] = yz - wx; out[10] = 1 - (xx + yy); out[11] = 0;
  out[12] = p[0]; out[13] = p[1]; out[14] = p[2]; out[15] = 1;
  return out;
}
function perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  out.fill(0);
  out[0] = f / aspect; out[5] = f; out[10] = (far + near) * nf; out[11] = -1; out[14] = 2 * far * near * nf;
  return out;
}
function lookAt(out, eye, center, up) {
  let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let l = 1 / Math.hypot(z0, z1, z2); z0 *= l; z1 *= l; z2 *= l;
  let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
  l = Math.hypot(x0, x1, x2) || 1; x0 /= l; x1 /= l; x2 /= l;
  const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
  out[15] = 1;
  return out;
}

const VERT = `
attribute vec3 aPos; attribute vec3 aNormal;
uniform mat4 uModel, uView, uProj;
varying vec3 vNormal; varying float vDepth;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vNormal = mat3(uModel) * aNormal;
  vec4 eye = uView * world;
  vDepth = -eye.z;
  gl_Position = uProj * eye;
}`;
const FRAG = `
precision mediump float;
varying vec3 vNormal; varying float vDepth;
uniform vec3 uColor; uniform vec3 uFog;
void main() {
  vec3 n = normalize(vNormal);
  float key = max(dot(n, normalize(vec3(0.45, 0.6, 0.85))), 0.0);
  float sky = 0.5 + 0.5 * n.z;
  vec3 lit = uColor * (0.34 + 0.20 * sky + 0.62 * key);
  lit += vec3(0.06) * pow(max(dot(n, normalize(vec3(-0.5, -0.3, 0.4))), 0.0), 2.0);
  float fog = clamp((vDepth - 1.2) / 3.0, 0.0, 1.0);
  gl_FragColor = vec4(mix(lit, uFog, fog * 0.55), 1.0);
}`;
const LINE_VERT = `
attribute vec3 aPos; uniform mat4 uView, uProj; varying float vDepth;
void main(){ vec4 e = uView * vec4(aPos,1.0); vDepth = -e.z; gl_Position = uProj * e; }`;
const LINE_FRAG = `
precision mediump float; varying float vDepth; uniform vec3 uColor; uniform vec3 uFog;
void main(){ float f = clamp((vDepth - 1.0)/3.0, 0.0, 1.0); gl_FragColor = vec4(mix(uColor, uFog, f), 1.0); }`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

function unitBox() {
  const f = [
    [[1, 0, 0], [1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
    [[-1, 0, 0], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]],
    [[0, 1, 0], [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]],
    [[0, -1, 0], [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],
    [[0, 0, 1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    [[0, 0, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]],
  ];
  const pos = [], nrm = [];
  for (const [n, a, b, c, d] of f) for (const v of [a, b, c, a, c, d]) { pos.push(...v); nrm.push(...n); }
  return { pos, nrm };
}

export async function createArenaRenderer(canvas, url) {
  const gl = canvas.getContext('webgl', { antialias: true, alpha: true });
  if (!gl) throw new Error('this browser has no WebGL');
  if (!gl.getExtension('OES_element_index_uint')) throw new Error('this browser cannot index past 65k vertices');

  const raw = await (await fetch(url)).arrayBuffer();
  const headerLen = new DataView(raw).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(raw, 4, headerLen)));
  let off = 4 + headerLen;
  const positions = new Float32Array(raw, off, meta.nvert * 3); off += meta.nvert * 12;
  const normals = new Float32Array(raw, off, meta.nvert * 3); off += meta.nvert * 12;
  const indices = new Uint32Array(raw, off, meta.nface * 3);

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  const nrmBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  const prog = compile(gl, VERT, FRAG);
  const u = n => gl.getUniformLocation(prog, n);
  const loc = { model: u('uModel'), view: u('uView'), proj: u('uProj'), color: u('uColor'), fog: u('uFog') };
  const aPos = gl.getAttribLocation(prog, 'aPos'), aNrm = gl.getAttribLocation(prog, 'aNormal');

  const lineProg = compile(gl, LINE_VERT, LINE_FRAG);
  const lLoc = { view: gl.getUniformLocation(lineProg, 'uView'), proj: gl.getUniformLocation(lineProg, 'uProj'),
                 color: gl.getUniformLocation(lineProg, 'uColor'), fog: gl.getUniformLocation(lineProg, 'uFog') };
  const lAPos = gl.getAttribLocation(lineProg, 'aPos');
  const S = 0.1, N = 40;
  const gridVerts = [];
  for (let i = -N; i <= N; i++) gridVerts.push(i * S, -N * S, 0, i * S, N * S, 0, -N * S, i * S, 0, N * S, i * S, 0);
  const gridMoved = new Float32Array(gridVerts.length);
  const gridBuf = gl.createBuffer();

  const box = unitBox();
  const boxPosBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, boxPosBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(box.pos), gl.STATIC_DRAW);
  const boxNrmBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, boxNrmBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(box.nrm), gl.STATIC_DRAW);
  const BOX_N = box.pos.length / 3;

  const proj = mat4(), view = mat4(), modelM = mat4(), localM = mat4(), bodyM = mat4(), laneM = mat4(), tmpM = mat4();
  let resolvedFor = null;
  function resolve(model) {
    if (resolvedFor === model) return;
    const byName = new Map();
    for (let b = 0; b < model.nbody; b++) byName.set(model.body(b).name, b);
    for (const d of meta.draws) { const b = byName.get(d.bodyName); if (b !== undefined) d.body = b; }
    resolvedFor = model;
  }

  return {
    triangles: meta.nface,
    /**
     * lanes: [{ data, laneY, colour: [r,g,b], root, visible }]
     * opts: { model, bg, grid, zoom, orbit }
     */
    render(lanes, opts) {
      resolve(opts.model);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const want = Math.min(devicePixelRatio || 1, 2);
      const MAX_PX = 6_000_000;
      const dpr = w * h * want * want > MAX_PX ? Math.max(1, Math.sqrt(MAX_PX / Math.max(w * h, 1))) : want;
      const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
      if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(opts.bg[0], opts.bg[1], opts.bg[2], 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // The camera frames the whole field: it follows the ducks' mean x, sits
      // behind and to the side, and backs off as more lanes are added so the
      // outermost duck stays in frame.
      // The camera frames where the ducks ARE, not where their lanes started: headings
      // drift, and a camera that tracked lane offsets lost the outermost duck off the edge.
      const shown = lanes.filter(l => l.visible);
      const px = shown.map(l => l.data.qpos[l.root]);
      const py = shown.map(l => l.data.qpos[l.root + 1] + l.laneY);
      const cx = px.length ? (Math.max(...px) + Math.min(...px)) / 2 : 0;
      const cy = py.length ? (Math.max(...py) + Math.min(...py)) / 2 : 0;
      const span = px.length ? Math.hypot(Math.max(...px) - Math.min(...px), Math.max(...py) - Math.min(...py)) : 0;
      const dist = (0.75 + 0.95 * span) / Math.min(Math.max(opts.zoom || 1, 0.4), 3);
      const yaw = opts.orbit || 0;
      const eye = [cx - dist * Math.cos(yaw), cy - dist * Math.sin(yaw) - 0.25 * dist, 0.45 * dist + 0.12];
      perspective(proj, 0.75, w / h, 0.02, 20);
      lookAt(view, eye, [cx, cy, 0.1], [0, 0, 1]);

      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(lLoc.view, false, view);
      gl.uniformMatrix4fv(lLoc.proj, false, proj);
      gl.uniform3fv(lLoc.color, opts.grid);
      gl.uniform3fv(lLoc.fog, opts.bg);
      const snapX = Math.round(cx / S) * S, snapY = Math.round(cy / S) * S;
      for (let i = 0; i < gridVerts.length; i += 3) {
        gridMoved[i] = gridVerts[i] + snapX; gridMoved[i + 1] = gridVerts[i + 1] + snapY; gridMoved[i + 2] = 0;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
      gl.bufferData(gl.ARRAY_BUFFER, gridMoved, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(lAPos);
      gl.vertexAttribPointer(lAPos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, gridMoved.length / 3);

      gl.useProgram(prog);
      gl.uniformMatrix4fv(loc.view, false, view);
      gl.uniformMatrix4fv(loc.proj, false, proj);
      gl.uniform3fv(loc.fog, opts.bg);
      gl.enableVertexAttribArray(aPos);
      gl.enableVertexAttribArray(aNrm);

      for (const lane of shown) {
        const data = lane.data;
        // The lane marker: a flat coloured strip under the duck, in the lane's
        // colour, so which duck is which never depends on reading a label.
        const rx = data.qpos[lane.root], ry = data.qpos[lane.root + 1];
        gl.bindBuffer(gl.ARRAY_BUFFER, boxPosBuf); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, boxNrmBuf); gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
        bodyM.fill(0);
        bodyM[0] = 0.14; bodyM[5] = 0.11; bodyM[10] = 0.0015; bodyM[15] = 1;
        bodyM[12] = rx; bodyM[13] = ry + lane.laneY; bodyM[14] = 0.0015;
        gl.uniformMatrix4fv(loc.model, false, bodyM);
        gl.uniform3fv(loc.color, lane.colour);
        gl.drawArrays(gl.TRIANGLES, 0, BOX_N);

        laneM[0] = laneM[5] = laneM[10] = laneM[15] = 1;
        laneM[1] = laneM[2] = laneM[3] = laneM[4] = laneM[6] = laneM[7] = laneM[8] = laneM[9] = laneM[11] = 0;
        laneM[12] = 0; laneM[13] = lane.laneY; laneM[14] = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf); gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        for (const d of meta.draws) {
          const b = d.body;
          if (b === undefined) continue;
          fromQuatPos(bodyM, [data.xquat[b * 4], data.xquat[b * 4 + 1], data.xquat[b * 4 + 2], data.xquat[b * 4 + 3]],
                      [data.xpos[b * 3], data.xpos[b * 3 + 1], data.xpos[b * 3 + 2]]);
          fromQuatPos(localM, d.quat, d.pos);
          multiply(tmpM, bodyM, localM);
          multiply(modelM, laneM, tmpM);
          gl.uniformMatrix4fv(loc.model, false, modelM);
          gl.uniform3f(loc.color, d.rgba[0], d.rgba[1], d.rgba[2]);
          const m = meta.meshes[d.mesh];
          gl.drawElements(gl.TRIANGLES, m.fn * 3, gl.UNSIGNED_INT, (m.f * 3) * 4);
        }
      }
    },
  };
}
