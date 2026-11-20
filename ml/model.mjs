export const LEGACY_LANDMARK_MODEL_VERSION = "landmark-ridge-1.0.0";
export const LANDMARK_MODEL_VERSION = "landmark-ridge-2.0.0";
export const LEGACY_FEATURE_COLUMNS = Array.from({ length: 12 }, (_, index) =>
  `radial_${String(index * 30).padStart(3, "0")}`,
);
export const FEATURE_COLUMNS = Array.from({ length: 24 }, (_, index) =>
  `radial_${String(index * 15).padStart(3, "0")}`,
);
export const TARGET_COLUMNS = [
  "d1_start_x", "d1_start_y", "d1_end_x", "d1_end_y",
  "d2_start_x", "d2_start_y", "d2_end_x", "d2_end_y",
];

function cleanContour(contour) {
  if (!Array.isArray(contour) || contour.length < 3 || contour.some((point) =>
    !Array.isArray(point) || point.length < 2 || !Number.isFinite(point[0]) || !Number.isFinite(point[1]))) {
    throw new Error("Сечение для ML-ориентиров должно содержать минимум три корректные точки.");
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of contour) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const epsilon = Math.max(Math.hypot(maxX - minX, maxY - minY) * 1e-10, 1e-12);
  const points = [];
  for (const point of contour) {
    const previous = points.at(-1);
    if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) > epsilon) {
      points.push([point[0], point[1]]);
    }
  }
  if (points.length > 2 && Math.hypot(points[0][0] - points.at(-1)[0], points[0][1] - points.at(-1)[1]) <= epsilon) {
    points.pop();
  }
  if (points.length < 3) throw new Error("Контур вырожден и не позволяет вычислить признаки ML.");
  return { points, epsilon };
}

function polygonCentroid(points, epsilon) {
  let twiceArea = 0, centerX = 0, centerY = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const cross = current[0] * next[1] - next[0] * current[1];
    twiceArea += cross;
    centerX += (current[0] + next[0]) * cross;
    centerY += (current[1] + next[1]) * cross;
  }
  if (Math.abs(twiceArea) <= epsilon * epsilon) {
    return points.reduce(
      (sum, point) => [sum[0] + point[0] / points.length, sum[1] + point[1] / points.length],
      [0, 0],
    );
  }
  return [centerX / (3 * twiceArea), centerY / (3 * twiceArea)];
}

function cross2(a, b) {
  return a[0] * b[1] - a[1] * b[0];
}

function rayPolygonRadius(points, center, direction, epsilon) {
  let maximum = 0;
  for (let index = 0; index < points.length; index += 1) {
    const first = [points[index][0] - center[0], points[index][1] - center[1]];
    const nextPoint = points[(index + 1) % points.length];
    const edge = [nextPoint[0] - points[index][0], nextPoint[1] - points[index][1]];
    const denominator = cross2(direction, edge);
    if (Math.abs(denominator) <= epsilon) {
      // A ray can coincide with a polygon edge. Include its forward endpoints.
      if (Math.abs(cross2(first, direction)) <= epsilon) {
        maximum = Math.max(maximum, first[0] * direction[0] + first[1] * direction[1]);
        const second = [nextPoint[0] - center[0], nextPoint[1] - center[1]];
        maximum = Math.max(maximum, second[0] * direction[0] + second[1] * direction[1]);
      }
      continue;
    }
    const distance = cross2(first, edge) / denominator;
    const edgeFraction = cross2(first, direction) / denominator;
    if (distance >= -epsilon && edgeFraction >= -epsilon && edgeFraction <= 1 + epsilon) {
      maximum = Math.max(maximum, distance);
    }
  }
  return Math.max(0, maximum);
}

/**
 * Version 2 radial signature. Radii are sampled by ray/polygon intersections,
 * not by selecting whichever STL vertices happen to fall into an angular bin.
 * Therefore the signature is much less sensitive to tessellation density.
 */
export function extractContourFeatures(contour) {
  const { points, epsilon } = cleanContour(contour);
  const center = polygonCentroid(points, epsilon);
  const maxRadius = points.reduce((maximum, point) =>
    Math.max(maximum, Math.hypot(point[0] - center[0], point[1] - center[1])), 0);
  if (!Number.isFinite(maxRadius) || maxRadius <= epsilon) {
    throw new Error("Контур не позволяет вычислить признаки для ML-модели.");
  }

  const angularStep = 2 * Math.PI / FEATURE_COLUMNS.length;
  const fallbackBins = [];
  const radii = FEATURE_COLUMNS.map((_, index) => {
    const angle = index * angularStep;
    const direction = [Math.cos(angle), Math.sin(angle)];
    let radius = rayPolygonRadius(points, center, direction, epsilon);
    if (radius <= epsilon) {
      // A concave outline can place its area centroid outside the polygon.
      // Keep geometry measurements available and flag this approximate bin.
      radius = Math.max(0, ...points.map((point) =>
        (point[0] - center[0]) * direction[0] + (point[1] - center[1]) * direction[1],
      ));
      fallbackBins.push(index);
    }
    return Math.min(1, radius / maxRadius);
  });
  if (radii.some((radius) => !Number.isFinite(radius))) {
    throw new Error("Контур содержит некорректные радиальные признаки для ML-модели.");
  }
  return { features: radii, center, maxRadius, fallbackBins };
}

/** Exact compatibility extractor for previously trained 12-bin models. */
export function extractLegacyContourFeatures(contour) {
  if (!Array.isArray(contour) || contour.length < 3) {
    throw new Error("Сечение для ML-ориентиров содержит слишком мало точек.");
  }
  const center = contour.reduce(
    (sum, point) => [sum[0] + point[0] / contour.length, sum[1] + point[1] / contour.length],
    [0, 0],
  );
  const normalizedRadii = Array(12).fill(0);
  const maxRadius = contour.reduce((max, point) =>
    Math.max(max, Math.hypot(point[0] - center[0], point[1] - center[1])), 0);
  if (maxRadius <= 0) throw new Error("Контур не позволяет вычислить признаки для ML-модели.");

  for (const point of contour) {
    const dx = point[0] - center[0], dy = point[1] - center[1];
    let angle = Math.atan2(dy, dx) * 180 / Math.PI;
    if (angle < 0) angle += 360;
    const bin = Math.round(angle / 30) % 12;
    normalizedRadii[bin] = Math.max(normalizedRadii[bin], Math.hypot(dx, dy) / maxRadius);
  }
  return { features: normalizedRadii, center, maxRadius };
}

function validateModel(model) {
  const isLegacy = model?.version === LEGACY_LANDMARK_MODEL_VERSION;
  const featureCount = isLegacy ? LEGACY_FEATURE_COLUMNS.length : FEATURE_COLUMNS.length;
  const isCurrent = model?.version === LANDMARK_MODEL_VERSION;
  if ((!isLegacy && !isCurrent) ||
      model.featureMeans?.length !== featureCount ||
      model.featureScales?.length !== featureCount ||
      model.coefficients?.length !== TARGET_COLUMNS.length ||
      model.intercepts?.length !== TARGET_COLUMNS.length ||
      model.coefficients.some((weights) => weights.length !== featureCount)) {
    throw new Error("ML-модель отсутствует или имеет неподдерживаемый формат. Переобучите модель для текущей версии.");
  }
  return { isLegacy, featureCount };
}

/** Accepts either a raw contour or a v2 {features, center, maxRadius} profile. */
export function predictLandmarks(contourOrProfile, model) {
  const { isLegacy, featureCount } = validateModel(model);
  let profile;
  if (!isLegacy && contourOrProfile && Array.isArray(contourOrProfile.features) &&
      contourOrProfile.features.length === featureCount && Array.isArray(contourOrProfile.center) &&
      Number.isFinite(contourOrProfile.maxRadius)) {
    profile = contourOrProfile;
  } else {
    if (!Array.isArray(contourOrProfile)) {
      throw new Error("Для старой ML-модели требуется исходный контур, для новой модели — радиальный профиль.");
    }
    profile = isLegacy
      ? extractLegacyContourFeatures(contourOrProfile)
      : extractContourFeatures(contourOrProfile);
  }
  const { features, center, maxRadius } = profile;
  if (features.length !== featureCount || !center.every(Number.isFinite) || maxRadius <= 0) {
    throw new Error("Радиальный профиль не соответствует размерности ML-модели.");
  }
  const predicted = model.coefficients.map((weights, output) => {
    let value = model.intercepts[output];
    for (let index = 0; index < features.length; index += 1) {
      const scale = model.featureScales[index];
      value += weights[index] * (features[index] - model.featureMeans[index]) / scale;
    }
    return value;
  });
  const points = [];
  for (let index = 0; index < predicted.length; index += 2) {
    points.push([
      Number((center[0] + predicted[index] * maxRadius).toFixed(5)),
      Number((center[1] + predicted[index + 1] * maxRadius).toFixed(5)),
    ]);
  }
  return {
    d1Start: points[0],
    d1End: points[1],
    d2Start: points[2],
    d2End: points[3],
  };
}
