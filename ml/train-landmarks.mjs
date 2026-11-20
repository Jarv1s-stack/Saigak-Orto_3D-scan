import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { FEATURE_COLUMNS, LEGACY_FEATURE_COLUMNS, LEGACY_LANDMARK_MODEL_VERSION, LANDMARK_MODEL_VERSION, TARGET_COLUMNS } from "./model.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_OUTPUT = resolve(ROOT, "data", "models", "landmark-model.json");

function parseCsvRow(line) {
  const cells = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && quoted && line[index + 1] === '"') {
      value += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      cells.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  if (quoted) throw new Error("В CSV-файле есть незакрытая кавычка.");
  cells.push(value);
  return cells;
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function solveLinearSystem(matrix, target) {
  const size = target.length;
  const a = matrix.map((row, index) => [...row, target[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-12) {
      throw new Error("Матрица обучения вырождена; требуются дополнительные и более разнообразные экспертные сканы.");
    }
    [a[column], a[pivot]] = [a[pivot], a[column]];
    const divisor = a[column][column];
    for (let cell = column; cell <= size; cell += 1) a[column][cell] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let cell = column; cell <= size; cell += 1) a[row][cell] -= factor * a[column][cell];
    }
  }
  return a.map((row) => row[size]);
}

function fitRidge(rows, lambda, featureColumns) {
  const featureMeans = featureColumns.map((_, column) =>
    rows.reduce((sum, row) => sum + row.features[column], 0) / rows.length,
  );
  const featureScales = featureColumns.map((_, column) => {
    const variance = rows.reduce((sum, row) => sum + (row.features[column] - featureMeans[column]) ** 2, 0) / rows.length;
    return Math.max(Math.sqrt(variance), 1e-6);
  });
  const design = rows.map((row) => [
    1,
    ...row.features.map((value, index) => (value - featureMeans[index]) / featureScales[index]),
  ]);
  const width = featureColumns.length + 1;
  const gram = Array.from({ length: width }, () => Array(width).fill(0));
  for (const vector of design) {
    for (let first = 0; first < width; first += 1) {
      for (let second = 0; second < width; second += 1) gram[first][second] += vector[first] * vector[second];
    }
  }
  for (let index = 1; index < width; index += 1) gram[index][index] += lambda;
  const coefficients = [];
  const intercepts = [];
  for (let target = 0; target < TARGET_COLUMNS.length; target += 1) {
    const values = rows.map((row) => row.targets[target]);
    const rhs = Array(width).fill(0);
    for (let row = 0; row < design.length; row += 1) {
      for (let column = 0; column < width; column += 1) rhs[column] += design[row][column] * values[row];
    }
    const weights = solveLinearSystem(gram, rhs);
    intercepts.push(weights[0]);
    coefficients.push(weights.slice(1));
  }
  return { featureMeans, featureScales, coefficients, intercepts };
}

function predict(row, model) {
  return model.coefficients.map((weights, output) =>
    model.intercepts[output] + weights.reduce((sum, weight, index) =>
      sum + weight * (row.features[index] - model.featureMeans[index]) / model.featureScales[index],
    0),
  );
}

function meanAbsoluteError(rows, predictions) {
  let absoluteError = 0;
  for (let row = 0; row < rows.length; row += 1) {
    for (let target = 0; target < TARGET_COLUMNS.length; target += 1) {
      absoluteError += Math.abs(predictions[row][target] - rows[row].targets[target]);
    }
  }
  return absoluteError / (rows.length * TARGET_COLUMNS.length);
}

function validateRows(lines) {
  const header = parseCsvRow(lines[0].replace(/^\uFEFF/, "")).map((value) => value.trim());
  const featureColumns = FEATURE_COLUMNS.every((name) => header.includes(name))
    ? FEATURE_COLUMNS
    : LEGACY_FEATURE_COLUMNS.every((name) => header.includes(name))
      ? LEGACY_FEATURE_COLUMNS
      : null;
  if (!featureColumns) {
    throw new Error(`CSV должен содержать полный набор признаков v2 (${FEATURE_COLUMNS.length} углов по 15°) или v1 (${LEGACY_FEATURE_COLUMNS.length} углов по 30°).`);
  }
  const expected = ["scan_code", ...featureColumns, ...TARGET_COLUMNS];
  const missing = expected.filter((name) => !header.includes(name));
  if (missing.length) throw new Error(`В заголовке CSV отсутствуют столбцы: ${missing.join(", ")}.`);
  const indexes = Object.fromEntries(expected.map((name) => [name, header.indexOf(name)]));
  const rows = [];
  const seenScanCodes = new Set();
  for (const [lineIndex, line] of lines.slice(1).entries()) {
    if (!line.trim()) continue;
    const cells = parseCsvRow(line);
    const scanCode = cells[indexes.scan_code]?.trim();
    if (!scanCode) throw new Error(`${lineIndex + 2}-жол: scan_code бос болмауы керек.`);
    if (seenScanCodes.has(scanCode)) {
      throw new Error(`${scanCode}: повторный scan_code нарушает правило «один скан на ребёнка».`);
    }
    seenScanCodes.add(scanCode);
    const features = featureColumns.map((name) => Number(cells[indexes[name]]));
    const targets = TARGET_COLUMNS.map((name) => Number(cells[indexes[name]]));
    if ([...features, ...targets].some((value) => !Number.isFinite(value))) {
      throw new Error(`Строка ${lineIndex + 2}: признаки и ориентиры должны быть числами.`);
    }
    if ([...features, ...targets].some((value) => value < -2 || value > 2)) {
      throw new Error(`Строка ${lineIndex + 2}: нормализованные признаки/ориентиры должны быть в диапазоне от −2 до 2.`);
    }
    rows.push({ scanCode, features, targets });
  }
  if (rows.length < 12) throw new Error("Требуется не менее 12 экспертно размеченных STL с уникальными кодами сканов.");
  return { rows, featureColumns };
}

function splitRows(rows, seed) {
  const shuffled = [...rows];
  const random = seededRandom(seed);
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
  }
  const testSize = Math.max(2, Math.ceil(rows.length * 0.2));
  return { train: shuffled.slice(testSize), test: shuffled.slice(0, testSize) };
}

function mainArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--input") options.input = argv[++index];
    else if (argv[index] === "--output") options.output = argv[++index];
    else if (argv[index] === "--seed") options.seed = Number(argv[++index]);
    else if (argv[index] === "--lambda") options.lambda = Number(argv[++index]);
    else throw new Error(`Неизвестный параметр: ${argv[index]}`);
  }
  if (!options.input) throw new Error("Использование: node ml/train-landmarks.mjs --input labels.csv [--output model.json] [--seed 42] [--lambda 0.1]");
  options.seed ??= 42;
  options.lambda ??= 0.1;
  if (!Number.isInteger(options.seed) || !Number.isFinite(options.lambda) || options.lambda <= 0) {
    throw new Error("--seed должен быть целым числом, а --lambda — положительным.");
  }
  options.output ??= DEFAULT_OUTPUT;
  return options;
}

try {
  const options = mainArguments(process.argv.slice(2));
  const lines = (await readFile(resolve(options.input), "utf8")).split(/\r?\n/);
  const { rows, featureColumns } = validateRows(lines);
  const isLegacy = featureColumns.length === LEGACY_FEATURE_COLUMNS.length;
  const { train, test } = splitRows(rows, options.seed);
  const model = fitRidge(train, options.lambda, featureColumns);
  const meanTargets = TARGET_COLUMNS.map((_, target) =>
    train.reduce((sum, row) => sum + row.targets[target], 0) / train.length,
  );
  const predictions = test.map((row) => predict(row, model));
  const baseline = test.map(() => meanTargets);
  const report = {
    dataset: { scans: rows.length, trainingScans: train.length, heldOutScans: test.length, groupingKey: "scan_code" },
    seed: options.seed,
    lambda: options.lambda,
    metric: "mean absolute error of normalized landmark x/y coordinates; not a clinical metric",
    ridgeMaeNormalized: meanAbsoluteError(test, predictions),
    meanBaselineMaeNormalized: meanAbsoluteError(test, baseline),
    testScanCodes: test.map((row) => row.scanCode).sort(),
  };
  const artifact = {
    version: isLegacy ? LEGACY_LANDMARK_MODEL_VERSION : LANDMARK_MODEL_VERSION,
    method: "standardized ridge regression with L2 regularization",
    featureDefinition: isLegacy
      ? "12 angular radial intersections at 30-degree intervals (legacy vertex-bin extractor)"
      : "24 polygon-ray intersections at 15-degree intervals; polygon-area centroid; normalized by maximum radius",
    featureColumns,
    targetColumns: TARGET_COLUMNS,
    featureMeans: model.featureMeans,
    featureScales: model.featureScales,
    coefficients: model.coefficients,
    intercepts: model.intercepts,
    training: report,
  };
  const output = resolve(options.output);
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(artifact, null, 2)}\n`, { flag: "w" });
  console.log(JSON.stringify({ output, ...report }, null, 2));
  if (report.ridgeMaeNormalized > report.meanBaselineMaeNormalized) {
    console.error("WARNING: ridge regression underperformed the mean baseline on the held-out split; do not use this model.");
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
