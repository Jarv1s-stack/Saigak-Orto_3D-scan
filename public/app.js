import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const elements = {
  fileInput: document.querySelector("#stl-input"),
  dropzone: document.querySelector("#dropzone"),
  uploadError: document.querySelector("#upload-error"),
  fileSummary: document.querySelector("#file-summary"),
  fileName: document.querySelector("#file-name"),
  fileMeta: document.querySelector("#file-meta"),
  qualityTag: document.querySelector("#quality-tag"),
  qualityBox: document.querySelector("#quality-box"),
  qualityTitle: document.querySelector("#quality-title"),
  qualityList: document.querySelector("#quality-list"),
  ageMonths: document.querySelector("#age-months"),
  reviewedCheckbox: document.querySelector("#reviewed-checkbox"),
  reviewSaveButton: document.querySelector("#review-save-button"),
  reviewStatus: document.querySelector("#review-status"),
  scale: document.querySelector("#scale-input"),
  slice: document.querySelector("#slice-input"),
  rotation: document.querySelector("#rotation-input"),
  angle: document.querySelector("#angle-input"),
  tiltX: document.querySelector("#tilt-x-input"),
  tiltY: document.querySelector("#tilt-y-input"),
  sliceValue: document.querySelector("#slice-value"),
  rotationValue: document.querySelector("#rotation-value"),
  angleValue: document.querySelector("#angle-value"),
  tiltXValue: document.querySelector("#tilt-x-value"),
  tiltYValue: document.querySelector("#tilt-y-value"),
  anteriorDirection: document.querySelector("#anterior-direction"),
  recalculate: document.querySelector("#recalculate-button"),
  viewerStage: document.querySelector("#viewer-stage"),
  viewerPlaceholder: document.querySelector("#viewer-placeholder"),
  scanIdentity: document.querySelector("#scan-identity"),
  scanCode: document.querySelector("#scan-code"),
  modelStatus: document.querySelector("#model-status"),
  legendAngle: document.querySelector("#legend-angle"),
  manualSelect: document.querySelector("#landmark-select"),
  clearLandmarks: document.querySelector("#clear-landmarks"),
  landmarkHint: document.querySelector("#landmark-hint"),
  exportPdf: document.querySelector("#export-pdf"),
  exportCsv: document.querySelector("#export-csv"),
  exportJson: document.querySelector("#export-json"),
};

let scan;
let renderer;
let camera;
let controls;
let scanRoot;
let scanGeometry;
let slicePlane;
let overlayGroup;
let manualMarks = [null, null, null, null];
let resizeObserver;
let settingsDirty = false;

function formatBytes(bytes) {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} МБ`
    : `${Math.max(1, Math.round(bytes / 1024))} КБ`;
}

function formatMetric(value, digits = 2) {
  return Number.isFinite(value)
    ? value.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : "—";
}

function showUploadError(message) {
  elements.uploadError.hidden = false;
  elements.uploadError.textContent = message;
}

function clearUploadError() {
  elements.uploadError.hidden = true;
  elements.uploadError.textContent = "";
}

async function responseJson(response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Ошибка сервера (${response.status}).`);
  return body;
}

async function uploadScan(file) {
  clearUploadError();
  if (!file.name.toLowerCase().endsWith(".stl")) {
    showUploadError("Выберите STL-файл 3D-модели.");
    return;
  }
  if (!file.size || file.size > 100 * 1024 * 1024) {
    showUploadError("Файл пуст или больше 100 МБ.");
    return;
  }
  elements.dropzone.classList.add("is-uploading");
  elements.dropzone.querySelector("strong").textContent = "Проверка модели…";
  const form = new FormData();
  form.append("model", file, file.name);
  form.append("scaleMmPerUnit", String(Number(elements.scale.value)));
  form.append("sliceFraction", String(Number(elements.slice.value) / 100));
  form.append("diagonalDegrees", String(Number(elements.angle.value)));
  form.append("tiltXDegrees", String(Number(elements.tiltX.value)));
  form.append("tiltYDegrees", String(Number(elements.tiltY.value)));
  form.append("anteriorDirection", elements.anteriorDirection.value);

  try {
    const response = await fetch("/api/scans", { method: "POST", body: form });
    scan = await responseJson(response);
    manualMarks = scan.metrics.manualLandmarks?.flat().map((point) => point) || [null, null, null, null];
    settingsDirty = false;
    elements.ageMonths.value = "";
    elements.reviewedCheckbox.checked = false;
    await renderScan(file);
    renderScanDetails();
    renderMeasurements();
    updateReviewControls();
    enableExports();
    elements.scanIdentity.hidden = false;
    elements.scanCode.textContent = scan.id;
    elements.recalculate.disabled = false;
  } catch (error) {
    showUploadError(error.message);
  } finally {
    elements.dropzone.classList.remove("is-uploading");
    elements.dropzone.querySelector("strong").textContent = "Перетащите STL-файл сюда";
  }
}

function renderScanDetails() {
  elements.fileSummary.hidden = false;
  elements.fileName.textContent = scan.fileName;
  elements.fileMeta.textContent =
    `${formatBytes(scan.fileBytes)} · ${scan.triangleCount.toLocaleString("ru-RU")} треугольников · методика ${scan.methodVersion}`;
  const quality = scan.quality;
  elements.qualityTag.textContent = quality.watertight ? "Топология замкнута" : "Нужна проверка";
  elements.qualityTag.classList.toggle("warning", !quality.watertight || !quality.valid);
  elements.qualityBox.hidden = false;
  elements.qualityBox.classList.toggle("warning", !quality.watertight || !quality.valid);
  elements.qualityTitle.textContent = quality.watertight && quality.valid
    ? "Базовая проверка STL пройдена"
    : "Качество STL требует внимания";
  elements.qualityList.replaceChildren();
  const checks = [
    `Треугольников: ${scan.triangleCount.toLocaleString("ru-RU")}`,
    `Граничных рёбер: ${quality.boundaryEdges.toLocaleString("ru-RU")} · несогласованных: ${quality.nonManifoldEdges.toLocaleString("ru-RU")}`,
    `${quality.completeness === "physician_review_required" ? "Полнота поверхности не определяется автоматически — нужна проверка врача." : "Полнота поверхности проверена."}`,
    ...quality.warnings,
  ];
  for (const [index, warning] of [...new Set(checks)].entries()) {
    const item = document.createElement("li");
    item.textContent = warning;
    elements.qualityList.append(item);
    if (index >= 4) break;
  }
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[group.children.length - 1];
    group.remove(child);
    child.geometry?.dispose();
    if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
    else child.material?.dispose();
  }
}

function setupViewer() {
  if (renderer) return true;
  const width = Math.max(1, elements.viewerStage.clientWidth);
  const height = Math.max(1, elements.viewerStage.clientHeight);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#101821");
  camera = new THREE.PerspectiveCamera(42, width / height, 0.01, 100000);
  camera.up.set(0, 0, 1);
  camera.position.set(0, -180, 110);
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  } catch {
    elements.viewerPlaceholder.querySelector("strong").textContent = "3D-просмотр недоступен в этом браузере";
    elements.viewerPlaceholder.querySelector("span").textContent = "Проверьте поддержку WebGL; расчёты и отчёт остаются доступны.";
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute("aria-label", "Интерактивная 3D-модель головы");
  elements.viewerStage.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight("#d9e8ed", "#25323a", 2));
  const keyLight = new THREE.DirectionalLight("#f4f3ec", 2.4);
  keyLight.position.set(-80, -110, 150);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight("#75cdb7", 1.1);
  fillLight.position.set(100, 40, 35);
  scene.add(fillLight);
  const grid = new THREE.GridHelper(180, 18, "#31504e", "#253b3d");
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -65;
  grid.material.transparent = true;
  grid.material.opacity = 0.34;
  scene.add(grid);
  scanRoot = new THREE.Group();
  scanRoot.userData.scene = scene;
  scene.add(scanRoot);
  overlayGroup = new THREE.Group();
  scene.add(overlayGroup);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.target.set(0, 0, 0);
  controls.update();
  resizeObserver = new ResizeObserver(() => {
    const newWidth = Math.max(1, elements.viewerStage.clientWidth);
    const newHeight = Math.max(1, elements.viewerStage.clientHeight);
    camera.aspect = newWidth / newHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(newWidth, newHeight);
  });
  resizeObserver.observe(elements.viewerStage);
  const animate = () => {
    if (!renderer) return;
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();
  return true;
}

function addLine(points, color, lineWidth = 1.6, loop = false) {
  if (points.length < 2) return;
  const vertices = points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, linewidth: lineWidth });
  const line = loop
    ? new THREE.LineLoop(geometry, material)
    : new THREE.Line(geometry, material);
  overlayGroup.add(line);
}

function updateOverlays() {
  if (!scan || !overlayGroup || !scanGeometry || !scan.meshCenter || !scan.meshBounds) return;
  clearGroup(overlayGroup);
  const metrics = scan.metrics;
  const scale = Number(elements.scale.value) || 1;
  const planeZ = metrics.planeZ - scan.meshCenter[2];
  const toScene = ([x, y]) => [x / scale - scan.meshCenter[0], y / scale - scan.meshCenter[1]];
  const contour = metrics.contour.map((point) => [...toScene(point), planeZ + 0.04]);
  addLine(contour, "#9ab5d0", 2, true);
  for (const [index, line] of metrics.diagonals.entries()) {
    addLine(line.map((point) => [...toScene(point), planeZ + 0.09]), index === 0 ? "#e6bd70" : "#df9e6d", 2.5);
  }
  const bounds = scan.meshBounds;
  const span = Math.max(bounds.x, bounds.y) * 0.62;
  addLine([[-span, 0, planeZ + 0.13], [span, 0, planeZ + 0.13]], "#df8d82");
  addLine([[0, -span, planeZ + 0.13], [0, span, planeZ + 0.13]], "#72d3b6");
  addLine([[0, 0, planeZ], [0, 0, planeZ + span]], "#7fa7e7");

  const addMarker = (point, color, size) => {
    if (!point) return;
    const [x, y] = toScene(point);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(size, 10, 8),
      new THREE.MeshBasicMaterial({ color }),
    );
    marker.position.set(x, y, planeZ + 0.18);
    overlayGroup.add(marker);
  };
  const predictions = metrics.predictedLandmarks;
  if (predictions) {
    const ordered = [predictions.d1Start, predictions.d1End, predictions.d2Start, predictions.d2End];
    ordered.forEach((point) => addMarker(point, "#70d7c0", Math.max(span * 0.018, 0.2)));
  }
  manualMarks.forEach((point) => addMarker(point, "#ffd27a", Math.max(span * 0.024, 0.25)));

  if (slicePlane) {
    overlayGroup.remove(slicePlane);
    slicePlane.geometry.dispose();
    slicePlane.material.dispose();
  }
  const planeSize = Math.max(bounds.x, bounds.y) * 1.12;
  const planeGeometry = new THREE.PlaneGeometry(planeSize, planeSize);
  const planeMaterial = new THREE.MeshBasicMaterial({
    color: "#7ea5c6", transparent: true, opacity: 0.11, side: THREE.DoubleSide, depthWrite: false,
  });
  slicePlane = new THREE.Mesh(planeGeometry, planeMaterial);
  slicePlane.position.z =
    (metrics.orientedBounds.minZ + metrics.orientedBounds.z * Number(elements.slice.value) / 100) - scan.meshCenter[2];
  overlayGroup.add(slicePlane);
}

async function renderScan(file) {
  if (!setupViewer()) return;
  const buffer = await file.arrayBuffer();
  scanGeometry = new STLLoader().parse(buffer);
  scanGeometry.computeBoundingBox();
  scanGeometry.computeVertexNormals();
  const center = scanGeometry.boundingBox.getCenter(new THREE.Vector3());
  scan.meshCenter = [center.x, center.y, center.z];
  scan.meshBounds = {
    x: scan.metrics.orientedBounds.x,
    y: scan.metrics.orientedBounds.y,
    z: scan.metrics.orientedBounds.z,
  };

  clearGroup(scanRoot);
  scanRoot.rotation.order = "XYZ";
  scanRoot.rotation.set(
    scan.metrics.tiltXDegrees * Math.PI / 180,
    scan.metrics.tiltYDegrees * Math.PI / 180,
    scan.metrics.orientationDegrees * Math.PI / 180,
  );
  const head = new THREE.Mesh(
    scanGeometry,
    new THREE.MeshStandardMaterial({ color: "#b7c7c1", roughness: 0.76, metalness: 0.02, side: THREE.DoubleSide }),
  );
  scanGeometry.translate(-center.x, -center.y, -center.z);
  head.castShadow = true;
  head.receiveShadow = true;
  scanRoot.add(head);
  updateOverlays();

  const extent = Math.max(...Object.values(scan.meshBounds));
  camera.position.set(extent * 1.8, -extent * 2.3, extent * 1.4);
  camera.near = Math.max(extent / 1000, 0.001);
  camera.far = Math.max(extent * 100, 1000);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
  elements.viewerPlaceholder.hidden = true;
}

function renderMeasurements() {
  if (!scan) return;
  const metrics = scan.metrics;
  document.querySelector("#metric-ci").textContent = `${formatMetric(metrics.ci)}%`;
  document.querySelector("#metric-cva").textContent = formatMetric(metrics.cva);
  document.querySelector("#metric-cvai").textContent = `${formatMetric(metrics.cvai)}%`;
  document.querySelector("#severity-value").textContent = metrics.severity;
  document.querySelector("#metric-dimensions").textContent =
    `${formatMetric(metrics.length)} × ${formatMetric(metrics.width)}`;
  document.querySelector("#metric-d1").textContent = formatMetric(metrics.diagonal1);
  document.querySelector("#metric-d2").textContent = formatMetric(metrics.diagonal2);
  document.querySelector("#metric-perimeter").textContent = formatMetric(metrics.perimeter);
  document.querySelector("#frontal-asymmetry").textContent =
    `${formatMetric(metrics.surfaceAsymmetry.frontalDifferenceMm)} мм`;
  document.querySelector("#occipital-asymmetry").textContent =
    `${formatMetric(metrics.surfaceAsymmetry.occipitalDifferenceMm)} мм`;
  const mapContainer = document.querySelector("#surface-map");
  mapContainer.replaceChildren();
  const maximumDifference = Math.max(...metrics.surfaceAsymmetry.pairs.map((pair) => pair.differenceMm), 0.001);
  for (const pair of metrics.surfaceAsymmetry.pairs) {
    const sector = document.createElement("div");
    sector.className = "surface-sector";
    const label = document.createElement("div");
    const directions = document.createElement("span");
    directions.textContent = pair.directions.join(" / ");
    const value = document.createElement("strong");
    value.textContent = `${formatMetric(pair.differenceMm)} мм`;
    label.append(directions, value);
    const track = document.createElement("div");
    track.className = "surface-track";
    const fill = document.createElement("div");
    fill.className = "surface-fill";
    fill.style.width = `${Math.max(3, pair.differenceMm / maximumDifference * 100)}%`;
    track.append(fill);
    sector.append(label, track);
    mapContainer.append(sector);
  }
  elements.sliceValue.textContent = `${elements.slice.value}%`;
  elements.rotationValue.textContent = `${Number(elements.rotation.value) > 0 ? "+" : ""}${elements.rotation.value}°`;
  elements.angleValue.textContent = `${elements.angle.value}°`;
  elements.tiltXValue.textContent = `${Number(elements.tiltX.value) > 0 ? "+" : ""}${elements.tiltX.value}°`;
  elements.tiltYValue.textContent = `${Number(elements.tiltY.value) > 0 ? "+" : ""}${elements.tiltY.value}°`;
  elements.legendAngle.textContent = `${elements.angle.value}°`;
  elements.clearLandmarks.disabled = manualMarks.every((point) => point === null);
  const missingPoints = manualMarks.filter((point) => point === null).length;
  elements.landmarkHint.textContent = settingsDirty
    ? "Сначала пересчитайте положение/сечение, затем задайте ручные точки."
    : missingPoints === 0
      ? "4 ручные точки заданы: D1/D2 пересчитаны по отмеченным точкам."
      : `${missingPoints} из 4 точек не отмечено. Бирюзовые точки — подсказки ML (если модель обучена).`;
  if (scanRoot) {
    const previewRotation = settingsDirty
      ? metrics.automaticOrientationDegrees + Number(elements.rotation.value)
      : metrics.orientationDegrees;
    scanRoot.rotation.z = previewRotation * Math.PI / 180;
    scanRoot.rotation.x = Number(elements.tiltX.value) * Math.PI / 180;
    scanRoot.rotation.y = Number(elements.tiltY.value) * Math.PI / 180;
    updateOverlays();
  }
}

function updateReviewControls() {
  if (!scan) return;
  const desiredAge = elements.ageMonths.value === "" ? null : Number(elements.ageMonths.value);
  const reviewChanged = elements.reviewedCheckbox.checked !== scan.reviewedByPhysician;
  const ageChanged = desiredAge !== scan.ageMonths;
  elements.reviewSaveButton.disabled = !reviewChanged && !ageChanged;
  elements.reviewSaveButton.querySelector("span:first-child").textContent =
    scan.reviewedByPhysician && !elements.reviewedCheckbox.checked ? "Убрать отметку врача" : "Сохранить отметку врача";
  elements.reviewStatus.textContent = scan.reviewedByPhysician ? "Проверено врачом" : "Ожидает проверки";
  elements.reviewStatus.classList.toggle("reviewed", scan.reviewedByPhysician);
}

function enableExports() {
  for (const [element, extension] of [
    [elements.exportPdf, "pdf"],
    [elements.exportCsv, "csv"],
    [elements.exportJson, "json"],
  ]) {
    element.href = `/api/scans/${encodeURIComponent(scan.id)}/report.${extension}`;
    element.setAttribute("aria-disabled", "false");
    element.setAttribute("download", `saigak-${scan.id}.${extension}`);
  }
}

async function recalculate() {
  if (!scan) return;
  elements.recalculate.disabled = true;
  elements.recalculate.textContent = "Пересчёт…";
  try {
    const response = await fetch(`/api/scans/${encodeURIComponent(scan.id)}/measurements`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scaleMmPerUnit: Number(elements.scale.value),
        sliceFraction: Number(elements.slice.value) / 100,
        orientationOffsetDegrees: Number(elements.rotation.value),
        diagonalDegrees: Number(elements.angle.value),
        tiltXDegrees: Number(elements.tiltX.value),
        tiltYDegrees: Number(elements.tiltY.value),
        anteriorDirection: elements.anteriorDirection.value,
        manualDiagonals: manualMarks.every((point) => point !== null)
          ? [[manualMarks[0], manualMarks[1]], [manualMarks[2], manualMarks[3]]]
          : undefined,
      }),
    });
    const previous = scan;
    const updated = await responseJson(response);
    scan = {
      ...updated,
      meshCenter: previous.meshCenter,
      meshBounds: {
        x: updated.metrics.orientedBounds.x,
        y: updated.metrics.orientedBounds.y,
        z: updated.metrics.orientedBounds.z,
      },
    };
    manualMarks = scan.metrics.manualLandmarks?.flat().map((point) => point) || [null, null, null, null];
    settingsDirty = false;
    scan.reviewedByPhysician = previous.reviewedByPhysician;
    scan.ageMonths = previous.ageMonths;
    renderMeasurements();
    renderScanDetails();
    updateReviewControls();
  } catch (error) {
    showUploadError(error.message);
  } finally {
    elements.recalculate.disabled = false;
    elements.recalculate.textContent = "Пересчитать измерения";
  }
}

async function saveReview() {
  if (!scan) return;
  if (elements.ageMonths.value !== "" &&
      (!Number.isInteger(Number(elements.ageMonths.value)) || Number(elements.ageMonths.value) < 0 || Number(elements.ageMonths.value) > 18)) {
    showUploadError("Возраст должен быть целым числом от 0 до 18 месяцев.");
    return;
  }
  elements.reviewSaveButton.disabled = true;
  try {
    const response = await fetch(`/api/scans/${encodeURIComponent(scan.id)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviewedByPhysician: elements.reviewedCheckbox.checked,
        ageMonths: elements.ageMonths.value === "" ? null : Number(elements.ageMonths.value),
      }),
    });
    const previous = scan;
    scan = { ...await responseJson(response), meshCenter: previous.meshCenter, meshBounds: previous.meshBounds };
    updateReviewControls();
    renderScanDetails();
  } catch (error) {
    showUploadError(error.message);
    elements.reviewSaveButton.disabled = false;
  }
}

elements.fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) uploadScan(file);
  elements.fileInput.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("drag-over");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("drag-over");
  });
}
elements.dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files?.[0];
  if (file) uploadScan(file);
});

elements.slice.addEventListener("input", () => {
  elements.sliceValue.textContent = `${elements.slice.value}%`;
  elements.recalculate.disabled = !scan;
  settingsDirty = Boolean(scan);
  if (scan) {
    elements.sliceValue.textContent = `${elements.slice.value}%`;
    updateOverlays();
    renderMeasurements();
  }
});
elements.rotation.addEventListener("input", () => {
  elements.rotationValue.textContent = `${Number(elements.rotation.value) > 0 ? "+" : ""}${elements.rotation.value}°`;
  settingsDirty = Boolean(scan);
  if (scanRoot) {
    scanRoot.rotation.set(
      Number(elements.tiltX.value) * Math.PI / 180,
      Number(elements.tiltY.value) * Math.PI / 180,
      (scan.metrics.automaticOrientationDegrees + Number(elements.rotation.value)) * Math.PI / 180,
    );
    renderMeasurements();
  }
});
elements.angle.addEventListener("input", () => {
  elements.angleValue.textContent = `${elements.angle.value}°`;
  elements.legendAngle.textContent = `${elements.angle.value}°`;
  settingsDirty = Boolean(scan);
});
for (const [control, output] of [[elements.tiltX, elements.tiltXValue], [elements.tiltY, elements.tiltYValue]]) {
  control.addEventListener("input", () => {
    output.textContent = `${Number(control.value) > 0 ? "+" : ""}${control.value}°`;
    settingsDirty = Boolean(scan);
    if (scanRoot) {
      scanRoot.rotation.set(
        Number(elements.tiltX.value) * Math.PI / 180,
        Number(elements.tiltY.value) * Math.PI / 180,
        (scan.metrics.automaticOrientationDegrees + Number(elements.rotation.value)) * Math.PI / 180,
      );
    }
  });
}
for (const control of [elements.scale, elements.slice, elements.rotation, elements.angle, elements.tiltX, elements.tiltY]) {
  control.addEventListener("input", () => {
    if (manualMarks.some((point) => point !== null)) {
      manualMarks = [null, null, null, null];
      renderMeasurements();
    }
    elements.recalculate.disabled = !scan;
  });
}
elements.anteriorDirection.addEventListener("change", () => {
  settingsDirty = Boolean(scan);
  elements.recalculate.disabled = !scan;
});
elements.ageMonths.addEventListener("input", updateReviewControls);
elements.reviewedCheckbox.addEventListener("change", updateReviewControls);
elements.recalculate.addEventListener("click", recalculate);
elements.reviewSaveButton.addEventListener("click", saveReview);
elements.clearLandmarks.addEventListener("click", () => {
  manualMarks = [null, null, null, null];
  renderMeasurements();
  recalculate();
});

function placeLandmark(event) {
  if (!scan || !renderer || settingsDirty || !scan.metrics?.contour?.length) {
    if (settingsDirty) elements.landmarkHint.textContent = "Сначала пересчитайте сечение, затем задайте ручные точки.";
    return;
  }
  const rectangle = renderer.domElement.getBoundingClientRect();
  const cursor = new THREE.Vector2(
    ((event.clientX - rectangle.left) / rectangle.width) * 2 - 1,
    -((event.clientY - rectangle.top) / rectangle.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(cursor, camera);
  const planeZ = scan.metrics.planeZ - scan.meshCenter[2];
  const intersection = raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -planeZ), new THREE.Vector3());
  if (!intersection) return;
  const scale = Number(elements.scale.value);
  const centerX = scan.meshCenter[0], centerY = scan.meshCenter[1];
  const cursorMm = [intersection.x * scale + centerX * scale, intersection.y * scale + centerY * scale];
  const nearest = scan.metrics.contour.reduce((closest, point) => {
    const distance = Math.hypot(point[0] - cursorMm[0], point[1] - cursorMm[1]);
    return distance < closest.distance ? { point, distance } : closest;
  }, { point: null, distance: Infinity });
  if (!nearest.point || nearest.distance > Math.max(scan.metrics.length, scan.metrics.width) * 0.3) return;
  const [lineIndex, endpointIndex] = elements.manualSelect.value.split("-").map(Number);
  const markIndex = lineIndex * 2 + endpointIndex;
  manualMarks[markIndex] = [...nearest.point];
  elements.clearLandmarks.disabled = false;
  renderMeasurements();
  if (manualMarks.every((point) => point !== null)) recalculate();
}

let pointerStart;
elements.viewerStage.addEventListener("pointerdown", (event) => {
  pointerStart = [event.clientX, event.clientY];
});
elements.viewerStage.addEventListener("pointerup", (event) => {
  if (!pointerStart || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) {
    pointerStart = null;
    return;
  }
  pointerStart = null;
  if (event.target === renderer?.domElement) placeLandmark(event);
});

try {
  const health = await fetch("/api/health").then(responseJson);
  if (health.trainedLandmarkModelAvailable) {
    elements.modelStatus.textContent = "Загружена обученная модель ориентиров; её подсказки всё равно должен проверить врач.";
  }
} catch (error) {
  showUploadError(`Не удалось подключиться к локальному серверу: ${error.message}`);
}
