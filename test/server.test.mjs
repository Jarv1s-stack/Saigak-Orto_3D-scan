import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CUBE_TRIANGLES = [
  [[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
  [[0, 0, 1], [1, 1, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
  [[0, 0, 0], [1, 0, 1], [1, 0, 0]], [[0, 0, 0], [0, 0, 1], [1, 0, 1]],
  [[0, 1, 0], [1, 1, 0], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 1]],
  [[0, 0, 0], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
  [[1, 0, 0], [1, 0, 1], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
];

function makeCubeStl() {
  const facets = CUBE_TRIANGLES.map((triangle) =>
    `facet normal 0 0 1\nouter loop\n${triangle.map((point) => `vertex ${point.join(" ")}`).join("\n")}\nendloop\nendfacet`,
  );
  return Buffer.from(`solid cube\n${facets.join("\n")}\nendsolid cube\n`);
}

async function findAvailablePort() {
  const server = createServer();
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const { port } = server.address();
  await new Promise((resolveClose, reject) => server.close((error) => error ? reject(error) : resolveClose()));
  return port;
}

test("local scan API uploads STL, recalculates, records review and exports reports", { timeout: 30_000 }, async () => {
  const temp = await mkdtemp(join(tmpdir(), "saigak-server-test-"));
  const port = await findAvailablePort();
  const server = spawn(process.execPath, ["server.mjs"], {
    cwd: ROOT,
    env: { ...process.env, SCAN_DATA_DIR: join(temp, "data"), PORT: String(port), HOST: "127.0.0.1" },
    stdio: "ignore",
  });
  const base = `http://127.0.0.1:${port}`;

  try {
    let health;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      try {
        const response = await fetch(`${base}/api/health`);
        health = await response.json();
        break;
      } catch {
        if (server.exitCode !== null) throw new Error(`Local server exited with ${server.exitCode}.`);
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 150));
      }
    }
    assert.equal(health?.ok, true);
    assert.equal(health.trainedLandmarkModelAvailable, false);

    const page = await fetch(base);
    const html = await page.text();
    assert.equal(page.status, 200);
    assert.match(page.headers.get("content-security-policy"), /nonce-/);
    assert.match(html, /Saigak-Orto/);
    assert.doesNotMatch(html, /__CSP_NONCE__/);
    for (const resource of [
      "/app.js",
      "/styles.css",
      "/vendor/three.module.js",
      "/vendor/three.core.js",
      "/vendor/loaders/STLLoader.js",
      "/vendor/controls/OrbitControls.js",
    ]) {
      assert.equal((await fetch(`${base}${resource}`)).status, 200, resource);
    }

    const stl = makeCubeStl();
    const form = new FormData();
    form.append("model", new Blob([stl]), "anonymous-cube.stl");
    const upload = await fetch(`${base}/api/scans`, { method: "POST", body: form });
    const scan = await upload.json();
    assert.equal(upload.status, 201);
    assert.equal(scan.quality.watertight, true);
    assert.equal(scan.metrics.ci, 100);
    assert.equal(scan.metrics.cva, 0);
    assert.equal(scan.metrics.cvai, 0);
    assert.equal(scan.metrics.surfaceAsymmetry.pairs.length, 12);
    assert.equal((await fetch(`${base}/api/scans/${scan.id}/mesh`)).headers.get("content-type"), "model/stl");

    const manualDiagonals = scan.metrics.diagonals;
    const measurementResponse = await fetch(`${base}/api/scans/${scan.id}/measurements`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scaleMmPerUnit: 1,
        sliceFraction: 0.5,
        diagonalDegrees: 30,
        orientationOffsetDegrees: 0,
        manualDiagonals,
      }),
    });
    const measured = await measurementResponse.json();
    assert.equal(measurementResponse.status, 200, JSON.stringify(measured));
    assert.ok(Math.abs(measured.metrics.perimeter - 4) < 1e-5);
    assert.deepEqual(measured.metrics.manualLandmarks, manualDiagonals);

    const scaled = await fetch(`${base}/api/scans/${scan.id}/measurements`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scaleMmPerUnit: 2, sliceFraction: 0.4, diagonalDegrees: 30, orientationOffsetDegrees: 0 }),
    }).then((response) => response.json());
    assert.ok(Math.abs(scaled.metrics.perimeter - 8) < 1e-5);
    const tilted = await fetch(`${base}/api/scans/${scan.id}/measurements`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scaleMmPerUnit: 1, sliceFraction: 0.5, diagonalDegrees: 30, tiltXDegrees: 15, tiltYDegrees: -10 }),
    }).then((response) => response.json());
    assert.equal(tilted.metrics.tiltXDegrees, 15);
    assert.equal(tilted.metrics.tiltYDegrees, -10);
    assert.ok(tilted.metrics.orientedBounds.z > 1);

    const reviewResponse = await fetch(`${base}/api/scans/${scan.id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewedByPhysician: true, ageMonths: 18 }),
    });
    const reviewed = await reviewResponse.json();
    assert.equal(reviewResponse.status, 200);
    assert.equal(reviewed.reviewedByPhysician, true);
    assert.equal(reviewed.ageMonths, 18);

    const jsonResponse = await fetch(`${base}/api/scans/${scan.id}/report.json`);
    const report = await jsonResponse.json();
    assert.ok(report.clinicalDisclaimer);
    assert.equal(report.measurements.surfaceAsymmetry.pairs.length, 12);
    assert.ok(Number.isFinite(report.measurements.ciPercent));
    assert.equal(report.formulae.asymmetrySeverity.length, 5);
    const csvResponse = await fetch(`${base}/api/scans/${scan.id}/report.csv`);
    assert.match(await csvResponse.text(), /"cvai_percent"/);
    const pdfResponse = await fetch(`${base}/api/scans/${scan.id}/report.pdf`);
    const pdf = Buffer.from(await pdfResponse.arrayBuffer());
    assert.equal(pdfResponse.status, 200);
    assert.equal(pdf.subarray(0, 4).toString(), "%PDF");
  } finally {
    server.kill();
    if (server.exitCode === null) await new Promise((resolveExit) => server.once("exit", resolveExit));
    await rm(temp, { recursive: true, force: true });
  }
});


test("server loads v2 24-feature weights and exposes ML landmark proposals", { timeout: 30_000 }, async () => {
  const temp = await mkdtemp(join(tmpdir(), "saigak-server-ml-v2-test-"));
  const dataDir = join(temp, "data");
  const modelDir = join(dataDir, "models");
  const port = await findAvailablePort();
  const featureColumns = Array.from({ length: 24 }, (_, index) => `radial_${String(index * 15).padStart(3, "0")}`);
  const targetColumns = [
    "d1_start_x", "d1_start_y", "d1_end_x", "d1_end_y",
    "d2_start_x", "d2_start_y", "d2_end_x", "d2_end_y",
  ];
  await mkdir(modelDir, { recursive: true });
  await writeFile(join(modelDir, "landmark-model.json"), JSON.stringify({
    version: "landmark-ridge-2.0.0",
    featureColumns,
    targetColumns,
    featureMeans: Array(24).fill(0.5),
    featureScales: Array(24).fill(0.1),
    coefficients: Array.from({ length: 8 }, () => Array(24).fill(0)),
    intercepts: Array(8).fill(0),
  }));
  const server = spawn(process.execPath, ["server.mjs"], {
    cwd: ROOT,
    env: { ...process.env, SCAN_DATA_DIR: dataDir, PORT: String(port), HOST: "127.0.0.1" },
    stdio: "ignore",
  });
  const base = `http://127.0.0.1:${port}`;

  try {
    let health;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      try {
        const response = await fetch(`${base}/api/health`);
        health = await response.json();
        break;
      } catch {
        if (server.exitCode !== null) throw new Error(`Local server exited with ${server.exitCode}.`);
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 150));
      }
    }
    assert.equal(health?.trainedLandmarkModelAvailable, true);
    assert.equal(health?.landmarkModelVersion, "landmark-ridge-2.0.0");
    const form = new FormData();
    form.append("model", new Blob([makeCubeStl()]), "anonymous-cube.stl");
    const upload = await fetch(`${base}/api/scans`, { method: "POST", body: form });
    const scan = await upload.json();
    assert.equal(upload.status, 201);
    assert.deepEqual(Object.keys(scan.metrics.predictedLandmarks), ["d1Start", "d1End", "d2Start", "d2End"]);
  } finally {
    server.kill();
    if (server.exitCode === null) await new Promise((resolveExit) => server.once("exit", resolveExit));
    await rm(temp, { recursive: true, force: true });
  }
});
