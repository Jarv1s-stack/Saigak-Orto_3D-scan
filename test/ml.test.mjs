import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { FEATURE_COLUMNS, LEGACY_FEATURE_COLUMNS, TARGET_COLUMNS, extractContourFeatures, predictLandmarks } from "../ml/model.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

test("extracts deterministic contour features and validates trained-model inference", () => {
  const contour = Array.from({ length: 48 }, (_, index) => {
    const angle = 2 * Math.PI * index / 48;
    return [4 * Math.cos(angle), 2 * Math.sin(angle)];
  });
  const extracted = extractContourFeatures(contour);
  assert.equal(extracted.features.length, FEATURE_COLUMNS.length);
  assert.ok(extracted.features.every((value) => value >= 0 && value <= 1));
  assert.ok(Math.abs(extracted.center[0]) < 1e-12);
  assert.ok(Math.abs(extracted.center[1]) < 1e-12);

  const legacyModel = {
    version: "landmark-ridge-1.0.0",
    featureMeans: Array(LEGACY_FEATURE_COLUMNS.length).fill(0),
    featureScales: Array(LEGACY_FEATURE_COLUMNS.length).fill(1),
    coefficients: TARGET_COLUMNS.map(() => Array(LEGACY_FEATURE_COLUMNS.length).fill(0)),
    intercepts: [0.1, 0.2, -0.1, 0.2, 0.1, -0.2, -0.1, -0.2],
  };
  const legacyLandmarks = predictLandmarks(contour, legacyModel);
  assert.deepEqual(Object.keys(legacyLandmarks), ["d1Start", "d1End", "d2Start", "d2End"]);
  assert.ok(Math.abs(legacyLandmarks.d1Start[0] - 0.4) < 1e-8);
  assert.ok(Math.abs(legacyLandmarks.d1Start[1] - 0.8) < 1e-8);

  const currentModel = {
    version: "landmark-ridge-2.0.0",
    featureMeans: Array(FEATURE_COLUMNS.length).fill(0),
    featureScales: Array(FEATURE_COLUMNS.length).fill(1),
    coefficients: TARGET_COLUMNS.map(() => Array(FEATURE_COLUMNS.length).fill(0)),
    intercepts: [0.1, 0.2, -0.1, 0.2, 0.1, -0.2, -0.1, -0.2],
  };
  const currentLandmarks = predictLandmarks(extracted, currentModel);
  assert.ok(Math.abs(currentLandmarks.d1Start[0] - 0.4) < 1e-8);
  assert.ok(Math.abs(currentLandmarks.d1Start[1] - 0.8) < 1e-8);
});

test("24-bin radial features are invariant to STL contour tessellation density", () => {
  const polygon = Array.from({ length: 20 }, (_, index) => {
    const angle = 2 * Math.PI * index / 20;
    return [7 * Math.cos(angle), 4 * Math.sin(angle)];
  });
  const dense = [];
  for (let index = 0; index < polygon.length; index += 1) {
    const a = polygon[index], b = polygon[(index + 1) % polygon.length];
    const subdivisions = index % 5 + 2;
    for (let step = 0; step < subdivisions; step += 1) {
      const t = step / subdivisions;
      dense.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  const first = extractContourFeatures(polygon);
  const second = extractContourFeatures(dense);
  assert.equal(first.features.length, 24);
  assert.equal(second.features.length, 24);
  assert.ok(first.features.every((value, index) => Math.abs(value - second.features[index]) < 1e-9));
  assert.ok(Math.abs(first.center[0] - second.center[0]) < 1e-9);
  assert.ok(Math.abs(first.center[1] - second.center[1]) < 1e-9);
});

test("training script makes a deterministic scan-level split and a model artifact", async () => {
  const temp = await mkdtemp(join(tmpdir(), "saigak-ml-test-"));
  try {
    const input = join(temp, "labels.csv");
    const firstModel = join(temp, "first.json");
    const secondModel = join(temp, "second.json");
    const header = ["scan_code", ...FEATURE_COLUMNS, ...TARGET_COLUMNS];
    const rows = [header.join(",")];
    for (let scan = 0; scan < 30; scan += 1) {
      const features = FEATURE_COLUMNS.map((_, feature) =>
        0.4 + 0.15 * Math.sin((scan + 1) * (feature + 2) * 0.117) +
        0.08 * Math.cos((scan + 3) * (feature + 1) * 0.071),
      );
      const targets = TARGET_COLUMNS.map((_, target) =>
        0.1 * Math.sin((scan + 1) * (target + 3) * 0.061),
      );
      rows.push([`anonymized-scan-${String(scan).padStart(3, "0")}`, ...features, ...targets].join(","));
    }
    await writeFile(input, `${rows.join("\n")}\n`);

    for (const output of [firstModel, secondModel]) {
      const result = spawnSync(process.execPath, [
        join(ROOT, "ml", "train-landmarks.mjs"), "--input", input,
        "--output", output, "--seed", "42", "--lambda", "0.1",
      ], { encoding: "utf8", cwd: ROOT });
      assert.equal(result.status, 0, result.stderr || result.stdout);
    }

    const trained = JSON.parse(await readFile(firstModel, "utf8"));
    const repeated = JSON.parse(await readFile(secondModel, "utf8"));
    assert.deepEqual(trained, repeated);
    assert.equal(trained.training.dataset.scans, 30);
    assert.equal(trained.training.dataset.trainingScans + trained.training.dataset.heldOutScans, 30);
    assert.equal(trained.training.metric, "mean absolute error of normalized landmark x/y coordinates; not a clinical metric");
    assert.ok(trained.training.ridgeMaeNormalized >= 0);
    assert.ok(trained.training.meanBaselineMaeNormalized >= 0);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});

test("trainer preserves compatibility with a legacy 12-feature annotation CSV", async () => {
  const temp = await mkdtemp(join(tmpdir(), "saigak-ml-v1-test-"));
  try {
    const input = join(temp, "legacy-labels.csv");
    const output = join(temp, "legacy-model.json");
    const header = ["scan_code", ...LEGACY_FEATURE_COLUMNS, ...TARGET_COLUMNS];
    const rows = [header.join(",")];
    for (let scan = 0; scan < 20; scan += 1) {
      const features = LEGACY_FEATURE_COLUMNS.map((_, feature) =>
        0.45 + 0.1 * Math.sin((scan + 1) * (feature + 1) * 0.12),
      );
      const targets = TARGET_COLUMNS.map((_, target) =>
        0.08 * Math.cos((scan + 2) * (target + 1) * 0.07),
      );
      rows.push([`legacy-scan-${scan}`, ...features, ...targets].join(","));
    }
    await writeFile(input, `${rows.join("\n")}\n`);
    const result = spawnSync(process.execPath, [
      join(ROOT, "ml", "train-landmarks.mjs"), "--input", input,
      "--output", output, "--seed", "42", "--lambda", "0.1",
    ], { encoding: "utf8", cwd: ROOT });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const trained = JSON.parse(await readFile(output, "utf8"));
    assert.equal(trained.version, "landmark-ridge-1.0.0");
    assert.equal(trained.featureColumns.length, 12);
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
});
