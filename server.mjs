import { createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { randomBytes, randomUUID } from "node:crypto";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import PDFDocument from "pdfkit";
import { measureScan, parseStl, ScanError } from "./lib/cranio.mjs";
import { LANDMARK_MODEL_VERSION, LEGACY_LANDMARK_MODEL_VERSION, predictLandmarks } from "./ml/model.mjs";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PUBLIC_DIR = join(ROOT, "public");
const DATA_DIR = resolve(process.env.SCAN_DATA_DIR || join(ROOT, "data"));
const SCANS_DIR = join(DATA_DIR, "clinical-scans");
const MODEL_PATH = join(DATA_DIR, "models", "landmark-model.json");
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || "127.0.0.1";
const MAX_FILE_BYTES = 100 * 1024 * 1024;
const MAX_REQUEST_BYTES = MAX_FILE_BYTES + 1024 * 1024;
const ALLOWED_STL_EXTENSIONS = new Set([".stl"]);
const scans = new Map();
let landmarkModel;
const FONT_PATHS = [
  process.env.REPORT_FONT_PATH,
  "C:\\Windows\\Fonts\\arial.ttf",
  "C:\\Windows\\Fonts\\calibri.ttf",
  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
].filter(Boolean);

const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function sendJson(response, status, value) {
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(JSON.stringify(value));
}

function readRequestBody(request) {
  return new Promise((resolveBody, reject) => {
    const chunks = [];
    let bytes = 0;
    let oversized = false;
    request.on("data", (chunk) => {
      if (oversized) return;
      bytes += chunk.length;
      if (bytes > MAX_REQUEST_BYTES) {
        oversized = true;
        reject(new HttpError(413, "Размер STL-файла не должен превышать 100 МБ."));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolveBody(Buffer.concat(chunks)));
    request.on("error", reject);
  });
}

function publicScan(scan) {
  const { mlFeatureProfile, ...clientMetrics } = scan.metrics;
  return {
    id: scan.id,
    fileName: scan.fileName,
    fileBytes: scan.mesh.fileBytes,
    triangleCount: scan.mesh.triangleCount,
    bounds: scan.mesh.bounds,
    quality: scan.mesh.quality,
    metrics: clientMetrics,
    ageMonths: scan.ageMonths,
    reviewedByPhysician: scan.reviewedByPhysician,
    methodVersion: "cranio-geometry-0.2.0",
    modelVersion: landmarkModel?.version || "landmark-ml-untrained",
  };
}

function addLandmarkPredictions(metrics) {
  if (!landmarkModel) return metrics;
  const predictionInput = landmarkModel.version === LANDMARK_MODEL_VERSION
    ? metrics.mlFeatureProfile
    : metrics.contour;
  return { ...metrics, predictedLandmarks: predictLandmarks(predictionInput, landmarkModel) };
}

function getScan(id) {
  const scan = scans.get(id);
  if (!scan) throw new HttpError(404, "Скан не найден. Если сервер был перезапущен, загрузите STL ещё раз.");
  return scan;
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError(400, "Некорректные данные JSON.");
  }
}

async function uploadScan(request, response) {
  const contentType = request.headers["content-type"] || "";
  if (!contentType.toLowerCase().startsWith("multipart/form-data;")) {
    throw new HttpError(400, "Отправьте STL-файл в формате multipart/form-data.");
  }

  const body = await readRequestBody(request);
  let form;
  try {
    form = await new Request("http://localhost", {
      method: "POST",
      headers: { "content-type": contentType },
      body,
    }).formData();
  } catch {
    throw new HttpError(400, "Не удалось прочитать форму. Выберите STL-файл ещё раз.");
  }

  const file = form.get("model");
  if (!file || typeof file === "string") throw new HttpError(400, "Выберите STL-файл 3D-модели головы.");
  if (file.size === 0 || file.size > MAX_FILE_BYTES) {
    throw new HttpError(413, "STL-файл пуст или превышает 100 МБ.");
  }
  const extension = extname(file.name).toLowerCase();
  if (!ALLOWED_STL_EXTENSIONS.has(extension)) {
    throw new HttpError(400, "Поддерживается только 3D-модель в формате STL.");
  }

  const bytes = Buffer.from(await file.arrayBuffer());
  const mesh = parseStl(bytes);
  const scaleMmPerUnit = Number(form.get("scaleMmPerUnit") || 1);
  const sliceFraction = Number(form.get("sliceFraction") || 0.5);
  const diagonalDegrees = Number(form.get("diagonalDegrees") || 30);
  const tiltXDegrees = Number(form.get("tiltXDegrees") || 0);
  const tiltYDegrees = Number(form.get("tiltYDegrees") || 0);
  const anteriorDirection = String(form.get("anteriorDirection") || "positive-y");
  const metrics = addLandmarkPredictions(measureScan(mesh, {
    scaleMmPerUnit, sliceFraction, diagonalDegrees, tiltXDegrees, tiltYDegrees, anteriorDirection,
  }));
  const id = randomUUID();
  const storagePath = join(SCANS_DIR, `${id}.stl`);
  await writeFile(storagePath, bytes, { flag: "wx" });
  const scan = {
    id,
    fileName: file.name,
    storagePath,
    mesh,
    metrics,
    scaleMmPerUnit,
    sliceFraction,
    diagonalDegrees,
    tiltXDegrees,
    tiltYDegrees,
    anteriorDirection,
    orientationOffsetDegrees: 0,
    ageMonths: null,
    reviewedByPhysician: false,
    createdAt: new Date().toISOString(),
  };
  scans.set(id, scan);
  sendJson(response, 201, publicScan(scan));
}

async function updateMeasurements(request, response, scan) {
  const body = parseJson((await readRequestBody(request)).toString("utf8"));
  const settings = {
    scaleMmPerUnit: body.scaleMmPerUnit,
    sliceFraction: body.sliceFraction,
    diagonalDegrees: body.diagonalDegrees,
    orientationOffsetDegrees: body.orientationOffsetDegrees,
    tiltXDegrees: body.tiltXDegrees,
    tiltYDegrees: body.tiltYDegrees,
    anteriorDirection: body.anteriorDirection,
    manualDiagonals: body.manualDiagonals,
  };
  const metrics = addLandmarkPredictions(measureScan(scan.mesh, settings));
  Object.assign(scan, settings, { metrics });
  sendJson(response, 200, publicScan(scan));
}

async function updateReview(request, response, scan) {
  const body = parseJson((await readRequestBody(request)).toString("utf8"));
  if (typeof body.reviewedByPhysician !== "boolean") {
    throw new HttpError(400, "Укажите, проверил ли врач результаты.");
  }
  if (body.ageMonths !== null && body.ageMonths !== undefined &&
      (!Number.isInteger(body.ageMonths) || body.ageMonths < 0 || body.ageMonths > 18)) {
    throw new HttpError(400, "Возраст должен быть целым числом от 0 до 18 месяцев.");
  }
  scan.reviewedByPhysician = body.reviewedByPhysician;
  scan.ageMonths = body.ageMonths ?? null;
  sendJson(response, 200, publicScan(scan));
}

function quoteCsv(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function reportRows(scan) {
  const { metrics } = scan;
  const { quality } = scan.mesh;
  return [
    ["scan_code", scan.id],
    ["age_months", scan.ageMonths ?? ""],
    ["physician_reviewed", scan.reviewedByPhysician ? "yes" : "no"],
    ["mesh_valid", quality.valid ? "yes" : "no"],
    ["mesh_watertight", quality.watertight ? "yes" : "no"],
    ["boundary_edges", quality.boundaryEdges],
    ["non_manifold_edges", quality.nonManifoldEdges],
    ["degenerate_triangles", quality.degenerateTriangles],
    ["completeness", quality.completeness],
    ["ci_percent", metrics.ci],
    ["cva_mm", metrics.cva],
    ["cvai_percent", metrics.cvai],
    ["asymmetry_severity", metrics.severity],
    ["head_length_mm", metrics.length],
    ["head_width_mm", metrics.width],
    ["diagonal_1_mm", metrics.diagonal1],
    ["diagonal_2_mm", metrics.diagonal2],
    ["section_perimeter_mm", metrics.perimeter],
    ["section_height_mm", metrics.sliceMm],
    ["section_diagnostics", JSON.stringify(metrics.sectionDiagnostics)],
    ["diagonal_angle_degrees", metrics.diagonalDegrees],
    ["orientation_correction_degrees", metrics.orientationDegrees],
    ["tilt_x_degrees", metrics.tiltXDegrees],
    ["tilt_y_degrees", metrics.tiltYDegrees],
    ["anterior_direction", metrics.anteriorDirection],
    ["frontal_asymmetry_mm_exploratory", metrics.surfaceAsymmetry.frontalDifferenceMm],
    ["occipital_asymmetry_mm_exploratory", metrics.surfaceAsymmetry.occipitalDifferenceMm],
    ["cross_section_radial_asymmetry_pairs_mm", JSON.stringify(metrics.surfaceAsymmetry.pairs)],
    ["ci_interpretation", metrics.ciInterpretation],
    ["ci_formula", "100 × W / L; no unapproved CI severity profile"],
    ["cva_formula", "|D1 − D2| mm"],
    ["cvai_formula", "100 × |D1 − D2| / max(D1, D2)"],
    ["cvai_severity_thresholds", "<3.5 below threshold; 3.5–<6.25 mild; 6.25–<8.75 moderate; 8.75–<11 severe; >=11 very severe"],
    ["manual_landmarks", metrics.manualLandmarks ? JSON.stringify(metrics.manualLandmarks) : "automatic contour endpoints"],
    ["method_version", "cranio-geometry-0.2.0"],
    ["model_version", landmarkModel?.version || "landmark-ml-untrained"],
    ["landmark_predictions", metrics.predictedLandmarks ? JSON.stringify(metrics.predictedLandmarks) : "not available"],
    ["clinical_disclaimer", "Research prototype; physician review required; not for diagnosis or treatment."],
  ];
}

function sendJsonReport(response, scan) {
  const report = {
    product: "Saigak-Orto",
    scanCode: scan.id,
    createdAt: scan.createdAt,
    ageMonths: scan.ageMonths,
    quality: scan.mesh.quality,
    physicianReviewed: scan.reviewedByPhysician,
    measurements: {
      ciPercent: scan.metrics.ci,
      cvaMm: scan.metrics.cva,
      cvaiPercent: scan.metrics.cvai,
      severity: scan.metrics.severity,
      headLengthMm: scan.metrics.length,
      headWidthMm: scan.metrics.width,
      diagonal1Mm: scan.metrics.diagonal1,
      diagonal2Mm: scan.metrics.diagonal2,
      sectionPerimeterMm: scan.metrics.perimeter,
      sectionHeightMm: scan.metrics.sliceMm,
      sectionDiagnostics: scan.metrics.sectionDiagnostics,
      diagonalAngleDegrees: scan.metrics.diagonalDegrees,
      orientationDegrees: scan.metrics.orientationDegrees,
      tiltXDegrees: scan.metrics.tiltXDegrees,
      tiltYDegrees: scan.metrics.tiltYDegrees,
      manualLandmarks: scan.metrics.manualLandmarks,
      surfaceAsymmetry: scan.metrics.surfaceAsymmetry,
    },
    formulae: {
      ci: "100 × width / length; CI thresholds are not universal and no severity category is assigned.",
      cva: "|diagonal1 − diagonal2| mm",
      cvai: "100 × |diagonal1 − diagonal2| / max(diagonal1, diagonal2)",
      asymmetrySeverity: [
        "CVAI < 3.5%: below scale threshold",
        "3.5% ≤ CVAI < 6.25%: mild",
        "6.25% ≤ CVAI < 8.75%: moderate",
        "8.75% ≤ CVAI < 11%: severe",
        "CVAI ≥ 11%: very severe",
      ],
      slice: "Robust triangle-plane intersection with scale-aware vertex/edge welding; largest unambiguous closed horizontal contour; principal axis estimated from uniformly arc-length-resampled boundary points.",
    },
    warnings: scan.mesh.quality.warnings,
    methodVersion: "cranio-geometry-0.2.0",
    modelVersion: landmarkModel?.version || "landmark-ml-untrained",
    proposedLandmarks: scan.metrics.predictedLandmarks ?? null,
    clinicalDisclaimer: "Research prototype for clinician-reviewed measurements only. Not a diagnostic device and not for treatment or helmet decisions.",
  };
  response.writeHead(200, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Disposition": `attachment; filename="saigak-${scan.id}.json"`,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(JSON.stringify(report, null, 2));
}

function sendCsvReport(response, scan) {
  const csv = ["field,value", ...reportRows(scan).map(([field, value]) => `${quoteCsv(field)},${quoteCsv(value)}`)].join("\r\n");
  response.writeHead(200, {
    "Content-Type": "text/csv; charset=utf-8",
    "Content-Disposition": `attachment; filename="saigak-${scan.id}.csv"`,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(`\uFEFF${csv}`);
}

function pdfLines(scan) {
  const metrics = scan.metrics;
  return [
    ["Scan code", scan.id],
    ["Age", scan.ageMonths === null ? "Not provided" : `${scan.ageMonths} months`],
    ["Physician review", scan.reviewedByPhysician ? "Marked reviewed" : "Not reviewed"],
    ["Mesh", `${scan.mesh.triangleCount} triangles; ${scan.mesh.quality.watertight ? "watertight" : "not watertight"}`],
    ["Mesh quality", scan.mesh.quality.valid ? "Basic integrity checks passed; completeness needs physician review." : "One or more integrity checks failed."],
    ["CI", `${metrics.ci.toFixed(2)}% — 100 × width / length; no CI severity profile is assigned.`],
    ["CVA", `${metrics.cva.toFixed(2)} mm — |D1 − D2|`],
    ["CVAI", `${metrics.cvai.toFixed(2)}% — 100 × |D1 − D2| / max(D1, D2)`],
    ["Asymmetry scale", metrics.severity],
    ["Head length / width", `${metrics.length.toFixed(2)} mm / ${metrics.width.toFixed(2)} mm`],
    ["Diagonal 1 / 2", `${metrics.diagonal1.toFixed(2)} mm / ${metrics.diagonal2.toFixed(2)} mm at ${metrics.diagonalDegrees}°`],
    ["Section perimeter / height", `${metrics.perimeter.toFixed(2)} mm / ${metrics.sliceMm.toFixed(2)} mm`],
    ["Section topology", `${metrics.sectionDiagnostics.closedLoopCount} closed loops; ${metrics.sectionDiagnostics.branchNodeCount} branch nodes; ${metrics.sectionDiagnostics.openNodeCount} open nodes`],
    ["Orientation", `X ${metrics.tiltXDegrees}°; Y ${metrics.tiltYDegrees}°; axial ${metrics.orientationDegrees}°`],
    ["Exploratory frontal / occipital contour asymmetry", `${metrics.surfaceAsymmetry.frontalDifferenceMm.toFixed(2)} / ${metrics.surfaceAsymmetry.occipitalDifferenceMm.toFixed(2)} mm; ${metrics.surfaceAsymmetry.coordinateConvention}`],
    ["Diagonal endpoint source", metrics.manualLandmarks ? "Manual points; physician review status shown above" : "Automatic contour endpoints; verify manually"],
    ["Method / model version", `cranio-geometry-0.2.0 / ${landmarkModel?.version || "landmark-ml-untrained"}`],
    ["Report generated", new Date().toISOString()],
  ];
}

function sendPdfReport(response, scan) {
  const fontPath = FONT_PATHS.find((path) => existsSync(path));
  if (!fontPath) throw new HttpError(503, "Не найден Unicode-шрифт для PDF. Укажите путь к TTF через REPORT_FONT_PATH.");

  const document = new PDFDocument({ size: "A4", margin: 48, info: { Title: `Saigak-Orto ${scan.id}`, Author: "Saigak-Orto" } });
  response.writeHead(200, {
    "Content-Type": "application/pdf",
    "Content-Disposition": `attachment; filename="saigak-${scan.id}.pdf"`,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  document.pipe(response);
  document.registerFont("Report", fontPath);
  document.font("Report").fontSize(18).text("Saigak-Orto", { continued: true });
  document.fontSize(10).fillColor("#677285").text("  |  Cranial scan measurement report");
  document.moveDown(0.35).fontSize(9).fillColor("#20283a").text(`Anonymous scan: ${scan.id}`);
  document.text(`Created: ${new Date().toISOString()}`);
  document.moveDown();

  for (const [label, value] of pdfLines(scan)) {
    document.fontSize(10).fillColor("#20283a").text(`${label}:`, { continued: true, lineGap: 3 });
    document.font("Report").fillColor("#465269").text(` ${value}`);
  }

  document.moveDown();
  document.fontSize(10).fillColor("#20283a").text("CVAI asymmetry scale:");
  document.fontSize(9).fillColor("#465269")
    .text("< 3.5% below threshold; 3.5–<6.25% mild; 6.25–<8.75% moderate; 8.75–<11% severe; >=11% very severe.");
  document.moveDown();
  document.fontSize(8).fillColor("#805323").text(`Quality notes: ${scan.mesh.quality.warnings.join(" ")}`);
  document.moveDown();
  document.fontSize(9).fillColor("#6f3440")
    .text("Research prototype only. The model cannot confirm anatomical completeness or provide a diagnosis. A qualified physician must review landmarks, measurements and image quality. Not for treatment or helmet decisions.");
  document.end();
}

async function serveStatic(pathname, response) {
  const routes = new Map([
    ["/", join(PUBLIC_DIR, "index.html")],
    ["/index.html", join(PUBLIC_DIR, "index.html")],
    ["/app.js", join(PUBLIC_DIR, "app.js")],
    ["/styles.css", join(PUBLIC_DIR, "styles.css")],
    ["/vendor/three.module.js", join(ROOT, "node_modules", "three", "build", "three.module.js")],
    ["/vendor/three.core.js", join(ROOT, "node_modules", "three", "build", "three.core.js")],
    ["/vendor/loaders/STLLoader.js", join(ROOT, "node_modules", "three", "examples", "jsm", "loaders", "STLLoader.js")],
    ["/vendor/controls/OrbitControls.js", join(ROOT, "node_modules", "three", "examples", "jsm", "controls", "OrbitControls.js")],
  ]);
  const filePath = routes.get(pathname);
  if (!filePath) throw new HttpError(404, "Запрашиваемая страница не найдена.");
  const nonce = randomBytes(18).toString("base64");
  response.writeHead(200, {
    "Content-Type": MIME_TYPES[extname(filePath)] || "text/javascript; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Security-Policy": `default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self' 'nonce-${nonce}'; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
  });
  if (pathname === "/" || pathname === "/index.html") {
    const html = await readFile(filePath, "utf8");
    response.end(html.replaceAll("__CSP_NONCE__", nonce));
    return;
  }
  createReadStream(filePath).pipe(response);
}

async function handleRequest(request, response) {
  const url = new URL(request.url, `http://${request.headers.host || "localhost"}`);
  if (request.method === "GET" && url.pathname === "/api/health") {
    sendJson(response, 200, {
      ok: true,
      service: "saigak-orto",
      localProcessing: true,
      trainedLandmarkModelAvailable: Boolean(landmarkModel),
      landmarkModelVersion: landmarkModel?.version || null,
      methodVersion: "cranio-geometry-0.2.0",
    });
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/scans") {
    await uploadScan(request, response);
    return;
  }

  const meshMatch = url.pathname.match(/^\/api\/scans\/([0-9a-f-]{36})\/mesh$/i);
  if (request.method === "GET" && meshMatch) {
    const scan = getScan(meshMatch[1]);
    response.writeHead(200, {
      "Content-Type": "model/stl",
      "Content-Length": scan.mesh.fileBytes,
      "Content-Disposition": `inline; filename="scan-${scan.id}.stl"`,
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    });
    createReadStream(scan.storagePath).pipe(response);
    return;
  }

  const scanRoute = url.pathname.match(/^\/api\/scans\/([0-9a-f-]{36})(?:\/(measurements|review|report\.(?:json|csv|pdf)))?$/i);
  if (scanRoute) {
    const scan = getScan(scanRoute[1]);
    const action = scanRoute[2];
    if (request.method === "GET" && !action) {
      sendJson(response, 200, publicScan(scan));
      return;
    }
    if (request.method === "POST" && action === "measurements") {
      await updateMeasurements(request, response, scan);
      return;
    }
    if (request.method === "POST" && action === "review") {
      await updateReview(request, response, scan);
      return;
    }
    if (request.method === "GET" && action === "report.json") {
      sendJsonReport(response, scan);
      return;
    }
    if (request.method === "GET" && action === "report.csv") {
      sendCsvReport(response, scan);
      return;
    }
    if (request.method === "GET" && action === "report.pdf") {
      sendPdfReport(response, scan);
      return;
    }
  }

  if (request.method === "GET" && (
    url.pathname === "/" || url.pathname === "/index.html" || url.pathname === "/app.js" ||
    url.pathname === "/styles.css" || url.pathname.startsWith("/vendor/")
  )) {
    await serveStatic(url.pathname, response);
    return;
  }
  throw new HttpError(404, "Запрашиваемая страница не найдена.");
}

await mkdir(SCANS_DIR, { recursive: true });
try {
  const parsedModel = JSON.parse(await readFile(MODEL_PATH, "utf8"));
  const expectedFeatureCount = parsedModel.version === LANDMARK_MODEL_VERSION ? 24
    : parsedModel.version === LEGACY_LANDMARK_MODEL_VERSION ? 12 : 0;
  if (!expectedFeatureCount || parsedModel.featureColumns?.length !== expectedFeatureCount ||
      parsedModel.targetColumns?.length !== 8) {
    throw new Error("Landmark model metadata or version does not match the supported schema.");
  }
  predictLandmarks([[0, 0], [1, 0], [0, 1]], parsedModel);
  landmarkModel = parsedModel;
  console.log(`Loaded local landmark model: ${landmarkModel.version}`);
} catch (error) {
  if (error.code !== "ENOENT") throw new Error(`Could not load landmark model at ${MODEL_PATH}: ${error.message}`);
}
const server = createServer((request, response) => {
  handleRequest(request, response).catch((error) => {
    if (response.headersSent || response.destroyed) return;
    const status = error instanceof ScanError || error instanceof HttpError ? error.status : 500;
    if (status === 500) console.error("Request failed:", error);
    sendJson(response, status, {
      error: status === 500 ? "Ошибка сервера. Подробности есть в журнале." : error.message,
    });
  });
});

server.listen(PORT, HOST, () => {
  console.log(`Прототип Saigak-Orto: http://localhost:${PORT}`);
  console.log("STL-файлы обрабатываются локально; внешние AI-сервисы не используются.");
});
