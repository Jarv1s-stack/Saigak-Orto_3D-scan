import { extractContourFeatures } from "../ml/model.mjs";

const MAX_TRIANGLES = 1_000_000;

export class ScanError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function calculateCranialIndices(width, length, diagonal1, diagonal2) {
  if (![width, length, diagonal1, diagonal2].every((value) => Number.isFinite(value) && value > 0)) {
    throw new ScanError(422, "Ширина, длина и обе диагонали должны быть положительными конечными числами.");
  }
  const cva = Math.abs(diagonal1 - diagonal2);
  return {
    ci: 100 * width / length,
    cva,
    cvai: 100 * cva / Math.max(diagonal1, diagonal2),
  };
}

function vertexKey(x, y, z, epsilon) {
  return `${Math.round(x / epsilon)},${Math.round(y / epsilon)},${Math.round(z / epsilon)}`;
}

function edgeKey(a, b) {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

export function parseStl(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 84) {
    throw new ScanError(400, "STL-файл пуст или повреждён.");
  }

  const declaredTriangles = buffer.readUInt32LE(80);
  const isBinary = 84 + declaredTriangles * 50 === buffer.length;
  let vertices;

  if (isBinary) {
    if (declaredTriangles === 0 || declaredTriangles > MAX_TRIANGLES) {
      throw new ScanError(400, `В STL-модели должно быть от 1 до ${MAX_TRIANGLES.toLocaleString("ru-RU")} треугольников.`);
    }
    vertices = new Float32Array(declaredTriangles * 9);
    for (let triangle = 0; triangle < declaredTriangles; triangle += 1) {
      const offset = 84 + triangle * 50 + 12;
      for (let coordinate = 0; coordinate < 9; coordinate += 1) {
        const value = buffer.readFloatLE(offset + coordinate * 4);
        if (!Number.isFinite(value))         throw new ScanError(400, "В STL-модели есть некорректная координата.");
        vertices[triangle * 9 + coordinate] = value;
      }
    }
  } else {
    const text = buffer.toString("utf8");
    const vertexPattern = /\bvertex\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)/gi;
    const values = [];
    for (const match of text.matchAll(vertexPattern)) {
      for (let axis = 1; axis <= 3; axis += 1) {
        const value = Number(match[axis]);
        if (!Number.isFinite(value)) throw new ScanError(400, "В ASCII STL есть некорректная координата.");
        values.push(value);
      }
      if (values.length > MAX_TRIANGLES * 9) {
        throw new ScanError(400, `В STL-модели допускается не более ${MAX_TRIANGLES.toLocaleString("ru-RU")} треугольников.`);
      }
    }
    if (values.length === 0 || values.length % 9 !== 0) {
      throw new ScanError(400, "ASCII STL неполный или не содержит корректных треугольников.");
    }
    vertices = Float32Array.from(values);
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let degenerateTriangles = 0;
  const vertexCount = vertices.length / 3;
  const epsilon = 1e-6;
  const vertexIds = new Int32Array(vertexCount);
  const uniqueVertices = new Map();
  const edgeUse = new Map();

  for (let index = 0; index < vertexCount; index += 1) {
    const x = vertices[index * 3];
    const y = vertices[index * 3 + 1];
    const z = vertices[index * 3 + 2];
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
    const key = vertexKey(x, y, z, epsilon);
    let id = uniqueVertices.get(key);
    if (id === undefined) {
      id = uniqueVertices.size;
      uniqueVertices.set(key, id);
    }
    vertexIds[index] = id;
  }

  for (let triangle = 0; triangle < vertexCount / 3; triangle += 1) {
    const offset = triangle * 9;
    const ax = vertices[offset], ay = vertices[offset + 1], az = vertices[offset + 2];
    const bx = vertices[offset + 3], by = vertices[offset + 4], bz = vertices[offset + 5];
    const cx = vertices[offset + 6], cy = vertices[offset + 7], cz = vertices[offset + 8];
    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;
    const crossX = aby * acz - abz * acy;
    const crossY = abz * acx - abx * acz;
    const crossZ = abx * acy - aby * acx;
    if (crossX * crossX + crossY * crossY + crossZ * crossZ < 1e-20) {
      degenerateTriangles += 1;
    }

    const first = vertexIds[triangle * 3];
    const second = vertexIds[triangle * 3 + 1];
    const third = vertexIds[triangle * 3 + 2];
    for (const [a, b] of [[first, second], [second, third], [third, first]]) {
      const key = edgeKey(a, b);
      edgeUse.set(key, (edgeUse.get(key) || 0) + 1);
    }
  }

  const bounds = [maxX - minX, maxY - minY, maxZ - minZ];
  if (bounds.some((length) => !Number.isFinite(length) || length <= 1e-6)) {
    throw new ScanError(400, "STL не содержит объёмной модели: размер по одной из осей равен нулю.");
  }

  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  for (const uses of edgeUse.values()) {
    if (uses === 1) boundaryEdges += 1;
    else if (uses !== 2) nonManifoldEdges += 1;
  }

  const closed = boundaryEdges === 0 && nonManifoldEdges === 0;
  const warnings = [];
  if (!closed) warnings.push("Поверхность не замкнута или рёбра не согласованы; периметр может быть неточным.");
  if (degenerateTriangles > 0) warnings.push(`Обнаружено вырожденных треугольников: ${degenerateTriangles}.`);
  warnings.push("Полноту лба, основания и затылка автоматически подтвердить нельзя — требуется проверка врача.");

  return {
    vertices,
    fileBytes: buffer.length,
    triangleCount: vertexCount / 3,
    bounds: { x: bounds[0], y: bounds[1], z: bounds[2], minX, minY, minZ, maxX, maxY, maxZ },
    pivot: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
    quality: {
      valid: degenerateTriangles === 0 && closed && bounds.every((length) => length > 0),
      watertight: closed,
      boundaryEdges,
      nonManifoldEdges,
      degenerateTriangles,
      completeness: "physician_review_required",
      warnings,
    },
  };
}

function sectionSegments(mesh, z) {
  const { vertices } = mesh;
  const diagonal = Math.hypot(mesh.bounds.x, mesh.bounds.y, mesh.bounds.z);
  // Float32 STL coordinates need a scale-aware merge tolerance. The value is
  // intentionally much smaller than any meaningful millimetre measurement.
  const epsilon = Math.max(diagonal * 1e-7, 1e-8);
  const nodes = [];
  const nodeCounts = [];
  const buckets = new Map();
  const segments = [];
  const segmentKeys = new Set();
  const coplanarEdges = new Map();
  let coplanarTriangleCount = 0;

  function bucketKey(x, y) {
    return `${Math.floor(x / epsilon)},${Math.floor(y / epsilon)}`;
  }

  function getNode(point) {
    const cellX = Math.floor(point[0] / epsilon);
    const cellY = Math.floor(point[1] / epsilon);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        const candidates = buckets.get(`${cellX + dx},${cellY + dy}`) || [];
        for (const id of candidates) {
          const existing = nodes[id];
          if (Math.hypot(existing[0] - point[0], existing[1] - point[1]) <= epsilon) {
            const count = nodeCounts[id];
            nodes[id] = [
              (existing[0] * count + point[0]) / (count + 1),
              (existing[1] * count + point[1]) / (count + 1),
            ];
            nodeCounts[id] += 1;
            return id;
          }
        }
      }
    }
    const id = nodes.length;
    nodes.push([point[0], point[1]]);
    nodeCounts.push(1);
    const key = bucketKey(point[0], point[1]);
    const cell = buckets.get(key) || [];
    cell.push(id);
    buckets.set(key, cell);
    return id;
  }

  function addSegment(firstPoint, secondPoint, target = segments) {
    const first = getNode(firstPoint);
    const second = getNode(secondPoint);
    if (first === second || Math.hypot(
      nodes[first][0] - nodes[second][0],
      nodes[first][1] - nodes[second][1],
    ) <= epsilon) return;
    const low = Math.min(first, second), high = Math.max(first, second);
    const key = `${low}:${high}`;
    if (target === segments) {
      if (segmentKeys.has(key)) return;
      segmentKeys.add(key);
      segments.push([first, second]);
    } else {
      target.set(key, (target.get(key) || 0) + 1);
    }
  }

  for (let offset = 0; offset < vertices.length; offset += 9) {
    const points = [
      [vertices[offset], vertices[offset + 1], vertices[offset + 2]],
      [vertices[offset + 3], vertices[offset + 4], vertices[offset + 5]],
      [vertices[offset + 6], vertices[offset + 7], vertices[offset + 8]],
    ];
    const distances = points.map((point) => point[2] - z);
    const signs = distances.map((distance) => Math.abs(distance) <= epsilon ? 0 : Math.sign(distance));

    if (signs.every((sign) => sign === 0)) {
      coplanarTriangleCount += 1;
      for (let edge = 0; edge < 3; edge += 1) {
        addSegment(points[edge], points[(edge + 1) % 3], coplanarEdges);
      }
      continue;
    }

    const hits = [];
    const pushUniqueHit = (point) => {
      if (!hits.some((existing) => Math.hypot(existing[0] - point[0], existing[1] - point[1]) <= epsilon)) {
        hits.push(point);
      }
    };

    for (let edge = 0; edge < 3; edge += 1) {
      const next = (edge + 1) % 3;
      const a = points[edge];
      const b = points[next];
      const signA = signs[edge];
      const signB = signs[next];
      if (signA === 0) pushUniqueHit([a[0], a[1]]);
      if (signB === 0) pushUniqueHit([b[0], b[1]]);
      if (signA * signB < 0) {
        const fraction = distances[edge] / (distances[edge] - distances[next]);
        pushUniqueHit([
          a[0] + fraction * (b[0] - a[0]),
          a[1] + fraction * (b[1] - a[1]),
        ]);
      }
    }
    if (hits.length === 2) addSegment(hits[0], hits[1]);
  }

  // The boundary of a coplanar triangle patch is the set of edges used once.
  // Internal coplanar triangulation edges are used twice and are discarded.
  for (const [key, uses] of coplanarEdges) {
    if (uses === 1 && !segmentKeys.has(key)) {
      const [first, second] = key.split(":").map(Number);
      const actualKey = `${Math.min(first, second)}:${Math.max(first, second)}`;
      if (!segmentKeys.has(actualKey)) {
        segmentKeys.add(actualKey);
        segments.push([first, second]);
      }
    }
  }

  const adjacency = Array.from({ length: nodes.length }, () => []);
  segments.forEach(([a, b], edge) => {
    adjacency[a].push(edge);
    adjacency[b].push(edge);
  });

  const visitedNodes = new Uint8Array(nodes.length);
  const visitedEdges = new Uint8Array(segments.length);
  const loops = [];
  let branchNodeCount = 0;
  let openNodeCount = 0;

  // Only degree-2 connected components form unambiguous closed section loops.
  // Branches/open chains are reported rather than stitched in arbitrary order.
  for (let start = 0; start < nodes.length; start += 1) {
    if (visitedNodes[start] || adjacency[start].length === 0) continue;
    const component = [];
    const stack = [start];
    visitedNodes[start] = 1;
    while (stack.length) {
      const node = stack.pop();
      component.push(node);
      for (const edge of adjacency[node]) {
        const [a, b] = segments[edge];
        const other = a === node ? b : a;
        if (!visitedNodes[other]) {
          visitedNodes[other] = 1;
          stack.push(other);
        }
      }
    }
    const degrees = component.map((node) => adjacency[node].length);
    branchNodeCount += degrees.filter((degree) => degree > 2).length;
    openNodeCount += degrees.filter((degree) => degree < 2).length;
    if (!degrees.every((degree) => degree === 2)) continue;

    const componentStart = component[0];
    const loop = [nodes[componentStart]];
    let current = componentStart;
    let previousEdge = -1;
    let closed = false;
    for (let step = 0; step <= component.length; step += 1) {
      const nextEdge = adjacency[current].find((edge) => edge !== previousEdge && !visitedEdges[edge]);
      if (nextEdge === undefined) break;
      visitedEdges[nextEdge] = 1;
      const [a, b] = segments[nextEdge];
      const nextNode = a === current ? b : a;
      if (nextNode === componentStart) {
        closed = true;
        break;
      }
      loop.push(nodes[nextNode]);
      previousEdge = nextEdge;
      current = nextNode;
    }
    if (closed && loop.length >= 3) loops.push(loop);
  }

  return {
    loops,
    diagnostics: {
      segmentCount: segments.length,
      nodeCount: nodes.length,
      closedLoopCount: loops.length,
      branchNodeCount,
      openNodeCount,
      coplanarTriangleCount,
      mergeTolerance: epsilon,
    },
  };
}

function tiltMesh(mesh, tiltXDegrees, tiltYDegrees) {
  if (tiltXDegrees === 0 && tiltYDegrees === 0) return mesh;
  const { bounds } = mesh;
  const center = mesh.pivot;
  const radiansX = tiltXDegrees * Math.PI / 180;
  const radiansY = tiltYDegrees * Math.PI / 180;
  const cosX = Math.cos(radiansX), sinX = Math.sin(radiansX);
  const cosY = Math.cos(radiansY), sinY = Math.sin(radiansY);
  const vertices = new Float32Array(mesh.vertices.length);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let index = 0; index < mesh.vertices.length; index += 3) {
    const x = mesh.vertices[index] - center[0];
    const y = mesh.vertices[index + 1] - center[1];
    const z = mesh.vertices[index + 2] - center[2];
    const tiltedY = y * cosX - z * sinX;
    const tiltedZ = y * sinX + z * cosX;
    const tiltedX = x * cosY + tiltedZ * sinY;
    const finalZ = -x * sinY + tiltedZ * cosY;
    const finalX = tiltedX + center[0];
    const finalY = tiltedY + center[1];
    const finalHeight = finalZ + center[2];
    vertices[index] = finalX;
    vertices[index + 1] = finalY;
    vertices[index + 2] = finalHeight;
    minX = Math.min(minX, finalX); maxX = Math.max(maxX, finalX);
    minY = Math.min(minY, finalY); maxY = Math.max(maxY, finalY);
    minZ = Math.min(minZ, finalHeight); maxZ = Math.max(maxZ, finalHeight);
  }

  return {
    ...mesh,
    vertices,
    pivot: center,
    bounds: {
      ...bounds,
      x: maxX - minX,
      y: maxY - minY,
      z: maxZ - minZ,
      minX, minY, minZ, maxX, maxY, maxZ,
    },
  };
}

function polygonArea(points) {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    area += current[0] * next[1] - next[0] * current[1];
  }
  return Math.abs(area) / 2;
}

function contourCentroid(points) {
  let twiceArea = 0, centerX = 0, centerY = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const cross = current[0] * next[1] - next[0] * current[1];
    twiceArea += cross;
    centerX += (current[0] + next[0]) * cross;
    centerY += (current[1] + next[1]) * cross;
  }
  if (Math.abs(twiceArea) < 1e-12) {
    return points.reduce(
      (sum, point) => [sum[0] + point[0] / points.length, sum[1] + point[1] / points.length],
      [0, 0],
    );
  }
  return [centerX / (3 * twiceArea), centerY / (3 * twiceArea)];
}

function resampleClosedContour(points, sampleCount = 512) {
  const edges = [];
  let perimeter = 0;
  for (let index = 0; index < points.length; index += 1) {
    const first = points[index];
    const second = points[(index + 1) % points.length];
    const length = Math.hypot(second[0] - first[0], second[1] - first[1]);
    if (length <= 1e-12) continue;
    edges.push({ first, second, start: perimeter, length });
    perimeter += length;
  }
  if (edges.length < 3 || perimeter <= 0) return points;
  const result = [];
  let edgeIndex = 0;
  for (let sample = 0; sample < sampleCount; sample += 1) {
    const distance = perimeter * sample / sampleCount;
    while (edgeIndex < edges.length - 1 && distance > edges[edgeIndex].start + edges[edgeIndex].length) {
      edgeIndex += 1;
    }
    const edge = edges[edgeIndex];
    const fraction = Math.max(0, Math.min(1, (distance - edge.start) / edge.length));
    result.push([
      edge.first[0] + fraction * (edge.second[0] - edge.first[0]),
      edge.first[1] + fraction * (edge.second[1] - edge.first[1]),
    ]);
  }
  return result;
}

function principalAxisDegrees(points) {
  // Equal arc-length samples avoid PCA bias caused by uneven STL triangle density.
  const center = contourCentroid(points);
  const samples = resampleClosedContour(points, Math.min(1024, Math.max(256, points.length)));
  let xx = 0, yy = 0, xy = 0;
  for (const point of samples) {
    const x = point[0] - center[0], y = point[1] - center[1];
    xx += x * x; yy += y * y; xy += x * y;
  }
  if (Math.hypot(xx - yy, 2 * xy) < 1e-12) return 0;
  return Math.atan2(2 * xy, xx - yy) * 90 / Math.PI;
}

function makeMeasurements(points, scale, sliceMm, orientationDegrees, diagonalDegrees, manualDiagonals, pivot) {
  const radians = orientationDegrees * Math.PI / 180;
  const cosine = Math.cos(radians), sine = Math.sin(radians);
  const rotated = points.map(([x, y]) => [
    (pivot[0] + (x - pivot[0]) * cosine - (y - pivot[1]) * sine) * scale,
    (pivot[1] + (x - pivot[0]) * sine + (y - pivot[1]) * cosine) * scale,
  ]);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of rotated) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const length = maxY - minY;
  const width = maxX - minX;
  if (length <= 0 || width <= 0) throw new ScanError(422, "В выбранном сечении недостаточно точек для измерений.");

  const angle = diagonalDegrees * Math.PI / 180;
  const getDiagonal = (direction) => {
    const dx = Math.sin(direction * angle);
    const dy = Math.cos(direction * angle);
    let minValue = Infinity, maxValue = -Infinity, minIndex = 0, maxIndex = 0;
    for (let index = 0; index < rotated.length; index += 1) {
      const [x, y] = rotated[index];
      const projection = x * dx + y * dy;
      if (projection < minValue) { minValue = projection; minIndex = index; }
      if (projection > maxValue) { maxValue = projection; maxIndex = index; }
    }
    return {
      length: maxValue - minValue,
      endpoints: [rotated[minIndex], rotated[maxIndex]],
    };
  };
  let firstDiagonal = getDiagonal(1);
  let secondDiagonal = getDiagonal(-1);
  if (manualDiagonals !== undefined) {
    if (!Array.isArray(manualDiagonals) || manualDiagonals.length !== 2 ||
        manualDiagonals.some((line) => !Array.isArray(line) || line.length !== 2 ||
          line.some((point) => !Array.isArray(point) || point.length !== 2 ||
            point.some((coordinate) => !Number.isFinite(coordinate))))) {
      throw new ScanError(400, "Формат ручных точек диагоналей некорректен.");
    }
    const tolerance = Math.max(width, length) * 0.08;
    for (const point of manualDiagonals.flat()) {
      const nearest = rotated.reduce((distance, outlinePoint) =>
        Math.min(distance, Math.hypot(point[0] - outlinePoint[0], point[1] - outlinePoint[1])),
      Infinity);
      if (nearest > tolerance) {
        throw new ScanError(400, "Отмеченная вручную точка слишком далеко от контура сечения.");
      }
    }
    const diagonalFromManualPoints = (line) => ({
      length: Math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]),
      endpoints: line,
    });
    firstDiagonal = diagonalFromManualPoints(manualDiagonals[0]);
    secondDiagonal = diagonalFromManualPoints(manualDiagonals[1]);
  }
  const indices = calculateCranialIndices(width, length, firstDiagonal.length, secondDiagonal.length);
  const perimeter = rotated.reduce((sum, point, index) => {
    const next = rotated[(index + 1) % rotated.length];
    return sum + Math.hypot(next[0] - point[0], next[1] - point[1]);
  }, 0);

  return {
    ...indices,
    length,
    width,
    diagonal1: firstDiagonal.length,
    diagonal2: secondDiagonal.length,
    perimeter,
    sliceMm,
    orientationDegrees,
    diagonalDegrees,
    manualLandmarks: manualDiagonals ?? null,
    contour: rotated,
    diagonals: [firstDiagonal.endpoints, secondDiagonal.endpoints],
  };
}

export function measureScan(mesh, options = {}) {
  const scale = Number(options.scaleMmPerUnit ?? 1);
  const sliceFraction = Number(options.sliceFraction ?? 0.5);
  const diagonalDegrees = Number(options.diagonalDegrees ?? 30);
  const orientationOffsetDegrees = Number(options.orientationOffsetDegrees ?? 0);
  const tiltXDegrees = Number(options.tiltXDegrees ?? 0);
  const tiltYDegrees = Number(options.tiltYDegrees ?? 0);
  const anteriorDirection = options.anteriorDirection ?? "positive-y";
  const manualDiagonals = options.manualDiagonals;
  if (!Number.isFinite(scale) || scale < 0.001 || scale > 100_000) {
    throw new ScanError(400, "Масштаб должен быть от 0,001 до 100000 мм на единицу.");
  }
  if (!Number.isFinite(sliceFraction) || sliceFraction < 0.1 || sliceFraction > 0.9) {
    throw new ScanError(400, "Высота сечения должна быть в диапазоне 10–90%.");
  }
  if (!Number.isFinite(diagonalDegrees) || diagonalDegrees < 15 || diagonalDegrees > 45) {
    throw new ScanError(400, "Угол диагоналей должен быть в диапазоне 15–45°.");
  }
  if (!Number.isFinite(orientationOffsetDegrees) || orientationOffsetDegrees < -180 || orientationOffsetDegrees > 180) {
    throw new ScanError(400, "Поправка ориентации должна быть в диапазоне от −180° до 180°.");
  }
  if (![tiltXDegrees, tiltYDegrees].every((angle) => Number.isFinite(angle) && angle >= -90 && angle <= 90)) {
    throw new ScanError(400, "Наклон по осям X/Y должен быть в диапазоне от −90° до 90°.");
  }
  if (!["positive-y", "negative-y"].includes(anteriorDirection)) {
    throw new ScanError(400, "Укажите переднее направление по оси +Y или −Y.");
  }

  const orientedMesh = tiltMesh(mesh, tiltXDegrees, tiltYDegrees);
  const sliceZ = orientedMesh.bounds.minZ + orientedMesh.bounds.z * sliceFraction;
  const section = sectionSegments(orientedMesh, sliceZ);
  const { loops, diagnostics: sectionDiagnostics } = section;
  if (loops.length === 0) {
    const detail = sectionDiagnostics.branchNodeCount || sectionDiagnostics.openNodeCount
      ? ` Найдено ветвлений: ${sectionDiagnostics.branchNodeCount}, открытых узлов: ${sectionDiagnostics.openNodeCount}.`
      : "";
    throw new ScanError(422, `На выбранной высоте не удалось построить однозначный замкнутый контур. Измените положение сечения или проверьте STL.${detail}`);
  }
  const outline = loops.reduce((largest, loop) => polygonArea(loop) > polygonArea(largest) ? loop : largest);
  const majorAxis = principalAxisDegrees(outline);
  const automaticOrientationDegrees = 90 - majorAxis;
  const orientationDegrees = automaticOrientationDegrees + orientationOffsetDegrees;
  const metrics = makeMeasurements(
    outline,
    scale,
    (sliceZ - orientedMesh.bounds.minZ) * scale,
    orientationDegrees,
    diagonalDegrees,
    manualDiagonals,
    orientedMesh.pivot,
  );
  const cvai = metrics.cvai;
  // The 24-bin profile is computed from the full-resolution section contour
  // before the API/UI contour is reduced for payload size.
  const radial = extractContourFeatures(metrics.contour);
  const pairs = radial.features.slice(0, 12).map((radius, index) => ({
    directions: [`${index * 15}°`, `${(index * 15 + 180) % 360}°`],
    differenceMm: Math.abs(radius - radial.features[index + 12]) * radial.maxRadius,
  }));
  // With +Y as anterior, the symmetric frontal rays are 60°/120° (4/8),
  // and the posterior rays are 240°/300° (16/20) on the 15° profile.
  const anteriorPair = anteriorDirection === "positive-y"
    ? [radial.features[4], radial.features[8]]
    : [radial.features[16], radial.features[20]];
  const posteriorPair = anteriorDirection === "positive-y"
    ? [radial.features[16], radial.features[20]]
    : [radial.features[4], radial.features[8]];
  const surfaceAsymmetry = {
    coordinateConvention: anteriorDirection === "positive-y" ? "anterior=+Y" : "anterior=−Y",
    exploratory: true,
    pairs,
    approximateDirections: radial.fallbackBins.map((index) => `${index * 15}°`),
    frontalDifferenceMm: Math.abs(anteriorPair[0] - anteriorPair[1]) * radial.maxRadius,
    occipitalDifferenceMm: Math.abs(posteriorPair[0] - posteriorPair[1]) * radial.maxRadius,
  };
  const severity = cvai < 3.5 ? "Ниже порога шкалы" :
    cvai < 6.25 ? "Лёгкая" :
    cvai < 8.75 ? "Умеренная" :
    cvai < 11 ? "Тяжёлая" : "Очень тяжёлая";

  return {
    ...metrics,
    automaticOrientationDegrees,
    orientationDegrees,
    tiltXDegrees,
    tiltYDegrees,
    anteriorDirection,
    surfaceAsymmetry,
    severity,
    ciInterpretation: "Пороговый профиль CI не утверждён; степень по CI не присваивается.",
    planeZ: sliceZ,
    orientedBounds: orientedMesh.bounds,
    sectionCount: loops.length,
    sectionDiagnostics: {
      ...sectionDiagnostics,
      selectedLoopPoints: outline.length,
      radialFallbackBins: radial.fallbackBins.length,
      selectedLoopAreaMm2: Number((polygonArea(outline) * scale * scale).toFixed(3)),
      perimeterMm: Number(metrics.perimeter.toFixed(3)),
      selection: "largest_closed_loop",
    },
    mlFeatureProfile: radial,
    contour: metrics.contour
      .filter((_, index) => index % Math.max(1, Math.ceil(metrics.contour.length / 720)) === 0)
      .map(([x, y]) => [Number(x.toFixed(5)), Number(y.toFixed(5))]),
    diagonals: metrics.diagonals.map((line) => line.map(([x, y]) => [Number(x.toFixed(5)), Number(y.toFixed(5))])),
  };
}
