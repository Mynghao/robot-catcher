// Bounce Lab front end: edits a config, sends it to the Python backend, draws what comes back.
// No geometry math happens here. Panels arrive fully resolved from webapp/sim_api.py.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // the simulator's world frame is z-up

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "bounce-lab-state-v1";
const COLORS = { ball: 0xeb6834, path: 0x2a78d6, obs: 0x23221f, zone: 0x1baf7a, muted: 0x6d6b63, other: 0x8a887f };

let init = null; // /api/init response
let state = null; // everything the user can edit; saved in localStorage
let results = null; // { request, times, runs, selected }
let lastPanels = null;
let hoveredBoard = null;

async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? undefined : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Can't reach the backend. Is webapp/server.py still running?");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function showError(message) {
  $("error").textContent = message || "";
  $("error").hidden = !message;
}

// ---------------------------------------------------------------- state

function defaultState() {
  return {
    source: "custom",
    config: structuredClone(init.config),
    throw: { mode: "sample", n: 12, min_bounces: 1, max_bounces: 2, p0: [3.0, 0.5, 1.5], v0: [-3.5, 0.2, 2.5] },
    seed: 0,
    t_end: 3.0,
    noise_std: 0.005,
  };
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && saved.config && saved.throw) return saved;
  } catch { /* unavailable or corrupt storage: fall through to defaults */ }
  return defaultState();
}

function saveState() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* optional convenience */ }
}

function buildRequest() {
  const isStage = state.source !== "custom";
  return {
    source: isStage ? "stage" : "custom",
    stage: isStage ? Number(state.source) : undefined,
    config: isStage ? undefined : state.config,
    throw: isStage ? { mode: "sample", n: state.throw.n } : state.throw,
    seed: state.seed,
    t_end: state.t_end,
    noise_std: state.noise_std,
  };
}

// ---------------------------------------------------------------- form

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function numberInput(obj, key, onChange, { step = 0.1, label } = {}) {
  const input = el("input", { type: "number", step, value: obj[key] ?? "" });
  if (label) input.setAttribute("aria-label", label);
  input.addEventListener("input", () => {
    const value = input.valueAsNumber;
    obj[key] = Number.isFinite(value) ? value : null; // the backend reports missing values
    input.classList.toggle("invalid", obj[key] === null);
    onChange();
  });
  return input;
}

function numberRow(label, obj, key, onChange, options) {
  return el("label", { className: "row" }, el("span", { textContent: label }), numberInput(obj, key, onChange, options));
}

function multiRow(label, obj, keys, onChange, options, axisLabels = ["x", "y", "z"]) {
  const row = el("div", { className: keys.length === 2 ? "triple pair" : "triple" });
  row.append(el("span", { className: "triple-label", textContent: label }));
  keys.forEach((key, i) => row.append(numberInput(obj, key, onChange, { ...options, label: `${label} ${axisLabels[i]}` })));
  return row;
}

function materialFields(target, onChange) {
  const select = el("select");
  for (const name of Object.keys(init.materials)) select.append(el("option", { value: name, textContent: name.replace("_", " ") }));
  const custom = el("option", { value: "custom", textContent: "custom" });
  select.append(custom);
  select.value = target.material in init.materials ? target.material : "custom";

  const elasticity = numberInput(target, "restitution", onChange, { step: 0.05, label: "elasticity" });
  const friction = numberInput(target, "friction", onChange, { step: 0.05, label: "friction" });
  select.addEventListener("change", () => {
    target.material = select.value;
    const preset = init.materials[select.value];
    if (preset) {
      target.restitution = preset.restitution;
      target.friction = preset.friction;
      elasticity.value = preset.restitution;
      friction.value = preset.friction;
      elasticity.classList.remove("invalid");
      friction.classList.remove("invalid");
    }
    onChange();
  });
  return [
    el("label", { className: "row" }, el("span", { textContent: "Material" }), select),
    el("label", { className: "row" }, el("span", { textContent: "Elasticity" }), elasticity),
    el("label", { className: "row" }, el("span", { textContent: "Friction" }), friction),
  ];
}

function buildGeometryForm() {
  const root = $("geometry");
  root.replaceChildren();
  const isStage = state.source !== "custom";
  root.hidden = isStage;
  $("stage-hint").hidden = !isStage;
  if (isStage) return;

  const config = state.config;
  const changed = geometryChanged;

  const room = el("section");
  room.append(el("h2", { textContent: "Room (metres)" }),
    multiRow("X min / max", config.room.x, [0, 1], changed, {}, ["min", "max"]),
    multiRow("Y min / max", config.room.y, [0, 1], changed, {}, ["min", "max"]),
    numberRow("Height", config.room, "height", changed));
  root.append(room);

  const surfaces = el("details");
  const surfacesBody = el("div", { className: "details-body" });
  for (const [name, cfg] of Object.entries(config.room.surfaces)) {
    surfacesBody.append(el("h3", { textContent: name.replace("_", " ") }), ...materialFields(cfg, changed));
  }
  surfaces.append(el("summary", { textContent: "Floor, ceiling and wall materials" }), surfacesBody);
  root.append(surfaces);

  const boards = el("section");
  boards.append(el("h2", { textContent: `Boards (${config.boards.length})` }));
  config.boards.forEach((board, i) => {
    const remove = el("button", { textContent: "Remove", type: "button" });
    remove.setAttribute("aria-label", `Remove board ${i + 1}`);
    remove.addEventListener("click", () => {
      config.boards.splice(i, 1);
      hoveredBoard = null;
      buildGeometryForm();
      geometryChanged();
    });
    const card = el("div", { className: "board" },
      el("div", { className: "board-title" }, el("span", { textContent: `Board ${i + 1}` }), remove),
      multiRow("Center", board.center, [0, 1, 2], changed),
      multiRow("Size w / h", board, ["width", "height"], changed, {}, ["width", "height"]),
      multiRow("Yaw / tilt °", board, ["yaw_deg", "tilt_deg"], changed, { step: 5 }, ["yaw", "tilt"]),
      ...materialFields(board, changed));
    const highlight = (on) => { hoveredBoard = on ? `board_${i}` : null; if (lastPanels) drawRoom(lastPanels); };
    card.addEventListener("focusin", () => highlight(true));
    card.addEventListener("focusout", () => highlight(false));
    card.addEventListener("mouseenter", () => highlight(true));
    card.addEventListener("mouseleave", () => highlight(false));
    boards.append(card);
  });
  const add = el("button", { textContent: "Add board", type: "button", className: "add-board" });
  add.addEventListener("click", () => {
    const preset = init.materials.plywood;
    config.boards.push({ center: [1.5, 0, 1.2], yaw_deg: 0, tilt_deg: 0, width: 1.2, height: 1.0,
      material: "plywood", restitution: preset.restitution, friction: preset.friction });
    buildGeometryForm();
    geometryChanged();
  });
  boards.append(add);
  root.append(boards);

  const region = el("details");
  region.append(el("summary", { textContent: "Throw region (where balls start)" }),
    el("div", { className: "details-body" },
      multiRow("Min", config.throw_region.lo, [0, 1, 2], changed),
      multiRow("Max", config.throw_region.hi, [0, 1, 2], changed)));
  root.append(region);
}

function buildThrowForm() {
  const root = $("throw-fields");
  root.replaceChildren();
  const t = state.throw;
  const changed = settingsChanged;
  const isStage = state.source !== "custom";
  $("throw-mode-row").hidden = isStage;
  $("throw-mode").value = t.mode;
  if (isStage) {
    const [lo, hi] = init.stages[state.source];
    root.append(numberRow("Throws", t, "n", changed, { step: 1 }),
      el("p", { className: "hint", textContent: lo === hi
        ? `This stage keeps throws with exactly ${lo} bounce${lo === 1 ? "" : "s"} before the catch zone.`
        : `This stage keeps throws with ${lo}–${hi} bounces before the catch zone.` }));
  } else if (t.mode === "sample") {
    root.append(numberRow("Throws", t, "n", changed, { step: 1 }),
      multiRow("Bounces min / max", t, ["min_bounces", "max_bounces"], changed, { step: 1 }, ["min", "max"]),
      el("p", { className: "hint", textContent: "Counts bounces before the ball first enters the catch zone." }));
  } else {
    root.append(multiRow("Start (m)", t.p0, [0, 1, 2], changed),
      multiRow("Velocity (m/s)", t.v0, [0, 1, 2], changed));
  }
  drawThrowPreview();
}

function buildRunForm() {
  $("run-fields").replaceChildren(
    numberRow("Seed", state, "seed", settingsChanged, { step: 1 }),
    numberRow("Duration (s)", state, "t_end", settingsChanged),
    numberRow("Noise σ (m)", state, "noise_std", settingsChanged, { step: 0.001 }));
}

function buildForms() {
  $("source").value = state.source;
  buildGeometryForm();
  buildThrowForm();
  buildRunForm();
}

// Results always describe the exact config that produced them, so any edit discards them.
// Re-validating also clears errors from a previous run.
let previewTimer = null;
let previewSeq = 0;
function geometryChanged() {
  saveState();
  clearResults();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 150);
}

function settingsChanged() {
  geometryChanged();
  drawThrowPreview();
}

async function refreshPreview() {
  const seq = ++previewSeq;
  if (state.source !== "custom") {
    showError("");
    $("run").disabled = false;
    clearRoom();
    setViewMessage("Procedural rooms are generated with each throw. Press Run simulation to see them.");
    return;
  }
  try {
    const preview = await api("/api/room", { config: state.config });
    if (seq !== previewSeq) return; // a newer edit is already on its way
    showError("");
    $("run").disabled = false;
    setViewMessage("");
    drawRoom(preview.panels);
    drawBounds(preview.bounds);
    drawRegion(preview.throw_region);
    drawThrowPreview();
    fitCameraOnce(preview.bounds);
  } catch (e) {
    if (seq !== previewSeq) return;
    showError(e.message);
    $("run").disabled = true;
  }
}

// ---------------------------------------------------------------- 3D scene

const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9e7e0);
scene.add(new THREE.AmbientLight(0xffffff, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(1, -2, 4);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

const roomGroup = new THREE.Group();
const boundsGroup = new THREE.Group();
const regionGroup = new THREE.Group();
const previewGroup = new THREE.Group();
const allPathsGroup = new THREE.Group();
const runGroup = new THREE.Group();
scene.add(roomGroup, boundsGroup, regionGroup, previewGroup, allPathsGroup, runGroup);

const ball = new THREE.Mesh(new THREE.SphereGeometry(0.05, 24, 16), new THREE.MeshStandardMaterial({ color: COLORS.ball, roughness: 0.5 }));
ball.visible = false;
scene.add(ball);

function disposeChildren(group) {
  group.traverse((node) => {
    node.geometry?.dispose();
    node.material?.dispose();
    node.dispose?.(); // helpers and instanced meshes
  });
  group.clear();
}

function drawStatics() {
  const zone = init.catch_zone;
  const shell = new THREE.Mesh(
    new THREE.SphereGeometry(zone.reach_max, 48, 32),
    new THREE.MeshBasicMaterial({
      color: COLORS.zone, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false,
      // Same cuts as catcher.throws.in_catch_zone: x >= 0 and z >= z_min.
      clippingPlanes: [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0), new THREE.Plane(new THREE.Vector3(0, 0, 1), -zone.z_min)],
    }));
  shell.position.fromArray(zone.shoulder);
  scene.add(shell);

  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, zone.shoulder[2], 24),
    new THREE.MeshStandardMaterial({ color: 0x55534c, roughness: 0.7 }));
  base.rotation.x = Math.PI / 2;
  base.position.set(0, 0, zone.shoulder[2] / 2);
  scene.add(base);
}

function colorFromRgba(rgba) {
  return new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
}

function panelMatrix(panel) {
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(...panel.u), new THREE.Vector3(...panel.v), new THREE.Vector3(...panel.normal));
  m.setPosition(new THREE.Vector3(...panel.center));
  return m;
}

function drawRoom(panels) {
  lastPanels = panels;
  disposeChildren(roomGroup);
  for (const panel of panels) {
    const isBoard = panel.name.startsWith("board");
    const [a, b] = panel.half_extents;
    const alpha = panel.name === "floor" ? 0.5 : panel.rgba[3]; // keep the floor from dominating the view
    const highlighted = panel.name === hoveredBoard;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2 * a, 2 * b), new THREE.MeshLambertMaterial({
      color: colorFromRgba(panel.rgba), side: THREE.DoubleSide, transparent: alpha < 1,
      opacity: alpha, depthWrite: alpha >= 1, emissive: highlighted ? COLORS.path : 0x000000, emissiveIntensity: 0.45,
    }));
    const matrix = panelMatrix(panel);
    mesh.applyMatrix4(matrix);
    roomGroup.add(mesh);
    const corners = [[-a, -b], [a, -b], [a, b], [-a, b]].map(([x, y]) => new THREE.Vector3(x, y, 0.001));
    const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(corners), new THREE.LineBasicMaterial({
      color: highlighted ? COLORS.path : isBoard ? 0x55534c : 0xa8a69c,
    }));
    edge.applyMatrix4(matrix);
    roomGroup.add(edge);
  }
}

function drawBounds(bounds) {
  disposeChildren(boundsGroup);
  const [x0, y0] = bounds.lo;
  const [x1, y1] = bounds.hi;
  const points = [];
  for (let x = Math.ceil(x0); x <= x1; x++) points.push(x, y0, 0.003, x, y1, 0.003);
  for (let y = Math.ceil(y0); y <= y1; y++) points.push(x0, y, 0.003, x1, y, 0.003);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  boundsGroup.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.14 })));
  const axes = new THREE.AxesHelper(0.5);
  axes.position.z = 0.005;
  boundsGroup.add(axes);
}

function drawRegion(region) {
  disposeChildren(regionGroup);
  if (!region) return;
  const box = new THREE.Box3(new THREE.Vector3(...region.lo), new THREE.Vector3(...region.hi));
  regionGroup.add(new THREE.Box3Helper(box, COLORS.path));
}

function drawThrowPreview() {
  disposeChildren(previewGroup);
  const t = state.throw;
  const isManual = state.source === "custom" && t.mode === "manual";
  regionGroup.visible = state.source === "custom" && !isManual;
  if (!isManual || results) return;
  if (![...t.p0, ...t.v0].every(Number.isFinite)) return;
  const v = new THREE.Vector3(...t.v0);
  const speed = v.length();
  const start = new THREE.Vector3(...t.p0);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.05, 20, 14), new THREE.MeshStandardMaterial({ color: COLORS.ball }));
  marker.position.copy(start);
  previewGroup.add(marker);
  if (speed > 0) previewGroup.add(new THREE.ArrowHelper(v.normalize(), start, 0.25 * speed, COLORS.ball, 0.12, 0.07));
}

function clearRoom() {
  lastPanels = null;
  disposeChildren(roomGroup);
  disposeChildren(boundsGroup);
  disposeChildren(regionGroup);
}

let fitted = false;
function fitCamera(bounds) {
  const lo = new THREE.Vector3(...bounds.lo);
  const hi = new THREE.Vector3(...bounds.hi);
  const center = lo.clone().add(hi).multiplyScalar(0.5);
  const size = hi.clone().sub(lo);
  const distance = Math.max(size.x, size.y, size.z) * 2.0;
  center.z = size.z * 0.35;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.1 * distance, -distance, 0.55 * distance));
  controls.update();
}

function fitCameraOnce(bounds) {
  if (fitted) return;
  fitted = true;
  fitCamera(bounds);
}

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

function setViewMessage(message) {
  $("view-message").textContent = message;
  $("view-message").hidden = !message;
}

// ---------------------------------------------------------------- run display

let travelledPath = null;
let observationPoints = null;

const Y_AXIS = new THREE.Vector3(0, 1, 0);

// One thin cylinder per sample interval. WebGL lines are stuck at 1 px, and changing
// `count` reveals the path as the ball travels.
function pathMesh(pos, color, radius = 0.012) {
  const mesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(radius, radius, 1, 8), new THREE.MeshBasicMaterial({ color }), pos.length - 1);
  const [start, end, delta, scale] = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  const rotation = new THREE.Quaternion();
  const matrix = new THREE.Matrix4();
  for (let i = 0; i < pos.length - 1; i++) {
    start.fromArray(pos[i]);
    end.fromArray(pos[i + 1]);
    delta.subVectors(end, start);
    const length = delta.length();
    rotation.setFromUnitVectors(Y_AXIS, length > 0 ? delta.divideScalar(length) : Y_AXIS);
    matrix.compose(start.add(end).multiplyScalar(0.5), rotation, scale.set(1, length, 1));
    mesh.setMatrixAt(i, matrix);
  }
  mesh.frustumCulled = false;
  return mesh;
}

// Drawn on top of everything: bounces lie on surfaces, which would hide half of a 3D marker.
let bounceTexture = null;
function bounceSprite(position) {
  if (!bounceTexture) {
    const c = el("canvas", { width: 64, height: 64 });
    const ctx = c.getContext("2d");
    ctx.arc(32, 32, 24, 0, 2 * Math.PI);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 12;
    ctx.strokeStyle = "#eb6834";
    ctx.stroke();
    bounceTexture = new THREE.CanvasTexture(c);
    bounceTexture.colorSpace = THREE.SRGBColorSpace;
  }
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: bounceTexture, depthTest: false, transparent: true }));
  sprite.position.fromArray(position);
  sprite.scale.setScalar(0.11);
  sprite.renderOrder = 5;
  return sprite;
}

function surfaceLabel(name) {
  const board = name.match(/^board_(\d+)$/);
  return board ? `Board ${Number(board[1]) + 1}` : name.replace("_", " ");
}

function lineFrom(flat, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(flat), 3));
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
}

function drawRun(run) {
  disposeChildren(runGroup);
  const flat = run.pos.flat();
  runGroup.add(lineFrom(flat, COLORS.path, 0.3));
  travelledPath = pathMesh(run.pos, COLORS.path);
  runGroup.add(travelledPath);

  const obsGeometry = new THREE.BufferGeometry();
  obsGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(run.obs.flat()), 3));
  observationPoints = new THREE.Points(obsGeometry, new THREE.PointsMaterial({ color: COLORS.obs, size: 0.022, depthTest: false, transparent: true }));
  observationPoints.renderOrder = 4; // on top of the path, which is thicker than typical noise
  observationPoints.visible = $("show-obs").checked;
  runGroup.add(observationPoints);

  for (const bounce of run.bounces) runGroup.add(bounceSprite(bounce.position));

  if (run.catch_index >= 0) {
    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.07), new THREE.MeshStandardMaterial({ color: COLORS.zone }));
    marker.position.fromArray(run.pos[run.catch_index]);
    runGroup.add(marker);
  }
}

function drawAllPaths() {
  disposeChildren(allPathsGroup);
  if (!results || !$("show-all").checked) return;
  results.runs.forEach((run, i) => {
    if (i !== results.selected) allPathsGroup.add(lineFrom(run.pos.flat(), COLORS.other, 0.45));
  });
}

function selectRun(index) {
  results.selected = index;
  const run = results.runs[index];
  drawRoom(run.panels);
  drawBounds(run.bounds);
  fitCameraOnce(run.bounds);
  drawRun(run);
  drawAllPaths();
  renderRunDetail(run);
  for (const button of $("run-list").querySelectorAll("button")) {
    button.setAttribute("aria-current", String(Number(button.dataset.index) === index));
  }
  $("run-list").querySelector(`[data-index="${index}"]`)?.scrollIntoView({ block: "nearest" });
  setPlayback(0, true);
}

function clearResults() {
  if (results) {
    results = null;
    disposeChildren(runGroup);
    disposeChildren(allPathsGroup);
    ball.visible = false;
    travelledPath = observationPoints = null;
    $("transport").hidden = true;
    $("results-body").hidden = true;
    $("results-empty").hidden = false;
    refreshPreview();
  }
}

function renderResults() {
  const { runs } = results;
  $("results-empty").hidden = true;
  $("results-body").hidden = false;
  $("transport").hidden = false;
  $("save-result").textContent = "";
  const sharedRoom = results.request.source !== "stage";
  $("show-all").disabled = !sharedRoom;
  if (!sharedRoom) $("show-all").checked = false;
  $("show-all-label").classList.toggle("disabled", !sharedRoom);
  $("show-all-label").title = sharedRoom ? "" : "Each throw here has its own room, so their paths can't be overlaid";

  const reached = runs.filter((r) => r.catch_index >= 0);
  const meanBounces = reached.length ? reached.reduce((sum, r) => sum + r.bounces_before_catch, 0) / reached.length : 0;
  $("results-title").textContent = `${runs.length} throw${runs.length === 1 ? "" : "s"}`;
  $("results-summary").textContent = reached.length
    ? `${results.times.length} samples each at ${init.obs_rate_hz} Hz. ${reached.length} reach the catch zone, after ${meanBounces.toFixed(1)} bounces on average.`
    : `${results.times.length} samples at ${init.obs_rate_hz} Hz. The ball never enters the catch zone.`;

  $("run-list").replaceChildren(...runs.map((run) => {
    const meta = run.catch_index >= 0
      ? `${run.bounces_before_catch} bounce${run.bounces_before_catch === 1 ? "" : "s"}, catch ${run.catch_time.toFixed(2)} s`
      : "misses catch zone";
    const button = el("button", { type: "button" },
      el("span", { textContent: `Throw ${run.index + 1}` }), el("span", { className: "meta", textContent: meta }));
    button.dataset.index = run.index;
    button.addEventListener("click", () => selectRun(run.index));
    return el("li", {}, button);
  }));

  const source = results.request.source === "stage" ? `stage${results.request.stage}` : "custom";
  $("save-name").value = `${source}_n${runs.length}_seed${results.request.seed}`;
  selectRun(0);
}

function renderRunDetail(run) {
  const root = $("run-detail");
  const fmt = (v, digits = 2) => v.map((x) => x.toFixed(digits)).join(", ");
  const dl = el("dl");
  const entries = [
    ["Start", `${fmt(run.p0)} m`],
    ["Velocity", `${fmt(run.v0)} m/s`],
    ["Bounces", `${run.bounces.length} in total${run.at_rest ? `, at rest at ${run.stop_time.toFixed(2)} s` : ""}`],
  ];
  for (const [term, value] of entries) dl.append(el("dt", { textContent: term }), el("dd", { textContent: value }));

  const children = [el("h2", { textContent: `Throw ${run.index + 1}` }), dl];
  if (results.request.source === "stage") {
    const edit = el("button", { textContent: "Edit this room", type: "button" });
    edit.addEventListener("click", () => {
      state.config = structuredClone(run.config);
      state.source = "custom";
      saveState();
      clearResults();
      buildForms();
      refreshPreview();
    });
    children.push(edit);
  }
  if (run.bounces.length) {
    const rows = run.bounces.map((b) => {
      const row = el("tr", {},
        el("td", { className: "num", textContent: b.t.toFixed(2) }),
        el("td", { textContent: surfaceLabel(b.surface) }),
        el("td", { className: "num", textContent: b.restitution.toFixed(2) }),
        el("td", { className: "num", textContent: `${b.speed_in.toFixed(1)} → ${b.speed_out.toFixed(1)}` }));
      row.title = "Jump to this bounce";
      row.addEventListener("click", () => setPlayback(b.t, false));
      return row;
    });
    children.push(el("h3", { textContent: "Bounces" }), el("table", {},
      el("thead", {}, el("tr", {},
        el("th", { className: "num", textContent: "t (s)" }), el("th", { textContent: "Surface" }),
        el("th", { className: "num", textContent: "e" }), el("th", { className: "num", textContent: "m/s" }))),
      el("tbody", {}, ...rows)));
  }
  root.replaceChildren(...children);
}

// ---------------------------------------------------------------- playback

const playback = { t: 0, playing: false, holdUntil: 0 };

function setPlayback(t, playing) {
  playback.t = t;
  playback.playing = playing;
  playback.holdUntil = 0;
  updatePlayButton();
}

function updatePlayButton() {
  $("play").textContent = playback.playing ? "❚❚" : "▶";
  $("play").setAttribute("aria-label", playback.playing ? "Pause" : "Play");
}

function updatePlaybackView() {
  if (!results) return;
  const { times } = results;
  const run = results.runs[results.selected];
  const end = times[times.length - 1];
  const t = Math.min(playback.t, end);
  const dt = times[1] - times[0];
  const k = Math.min(Math.floor(t / dt + 1e-9), times.length - 1);
  const next = Math.min(k + 1, times.length - 1);
  const a = run.pos[k];
  const b = run.pos[next];
  const f = next > k ? (t - times[k]) / dt : 0;
  ball.position.set(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f);
  ball.visible = true;
  if (travelledPath) travelledPath.count = k;
  observationPoints?.geometry.setDrawRange(0, k + 1);

  $("scrub").max = end;
  if (document.activeElement !== $("scrub") || playback.playing) $("scrub").value = t;
  const bounced = run.bounces.filter((bounce) => bounce.t <= t).length;
  $("readout").textContent = `t ${t.toFixed(2)} s · ${run.speed[k].toFixed(1)} m/s · height ${a[2].toFixed(2)} m · ${bounced} bounce${bounced === 1 ? "" : "s"}`;
}

let lastFrame = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  const elapsed = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  if (results && playback.playing) {
    const end = results.times[results.times.length - 1];
    if (playback.t >= end) {
      playback.holdUntil ||= now + 800; // rest on the last frame, then loop
      if (now >= playback.holdUntil) { playback.t = 0; playback.holdUntil = 0; }
    } else {
      playback.t = Math.min(end, playback.t + elapsed * Number($("speed").value));
    }
  }
  updatePlaybackView();
  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- actions

async function run() {
  $("run").disabled = true;
  $("run").textContent = "Running…";
  showError("");
  const request = buildRequest();
  try {
    const response = await api("/api/simulate", request);
    if (JSON.stringify(request) !== JSON.stringify(buildRequest())) return; // edited while running
    results = { request: structuredClone(request), times: response.times, runs: response.runs, selected: 0 };
    disposeChildren(previewGroup);
    setViewMessage("");
    renderResults();
  } catch (e) {
    showError(e.message);
  } finally {
    $("run").disabled = false;
    $("run").textContent = "Run simulation";
  }
}

async function save() {
  $("save").disabled = true;
  $("save-result").textContent = "Saving…";
  try {
    const saved = await api("/api/save", { run: results.request, name: $("save-name").value });
    $("save-result").textContent = `Saved ${saved.throws} throw${saved.throws === 1 ? "" : "s"} to ${saved.path} (${(saved.bytes / 1024).toFixed(0)} KB).`;
  } catch (e) {
    $("save-result").textContent = e.message;
  } finally {
    $("save").disabled = false;
  }
}

function wireEvents() {
  $("run").addEventListener("click", run);
  $("save").addEventListener("click", save);
  $("reset").addEventListener("click", () => {
    state = defaultState();
    saveState();
    clearResults();
    buildForms();
    refreshPreview();
  });
  $("source").addEventListener("change", () => {
    state.source = $("source").value;
    saveState();
    clearResults();
    buildForms();
    refreshPreview();
  });
  $("throw-mode").addEventListener("change", () => {
    state.throw.mode = $("throw-mode").value;
    buildThrowForm();
    settingsChanged();
  });
  $("play").addEventListener("click", () => {
    playback.playing = !playback.playing;
    if (playback.playing && results && playback.t >= results.times[results.times.length - 1]) playback.t = 0;
    updatePlayButton();
  });
  $("scrub").addEventListener("input", () => {
    setPlayback($("scrub").valueAsNumber, false);
  });
  $("show-all").addEventListener("change", drawAllPaths);
  $("show-obs").addEventListener("change", () => {
    if (observationPoints) observationPoints.visible = $("show-obs").checked;
  });
  $("reset-view").addEventListener("click", () => {
    const bounds = results ? results.runs[results.selected].bounds : null;
    if (bounds) return fitCamera(bounds);
    const b = state.config.room;
    if (state.source === "custom") fitCamera({ lo: [b.x[0], b.y[0], 0], hi: [b.x[1], b.y[1], b.height] });
  });
}

async function start() {
  try {
    init = await api("/api/init");
  } catch (e) {
    setViewMessage(e.message);
    return;
  }
  state = loadState();
  drawStatics();
  buildForms();
  wireEvents();
  resize();
  await refreshPreview();
  requestAnimationFrame(animate);
}

start();
