import test from "node:test";
import assert from "node:assert/strict";
import { calculateCranialIndices, measureScan, parseStl, ScanError } from "../lib/cranio.mjs";

const cubeTriangles = [
  [[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
  [[0, 0, 1], [1, 1, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
  [[0, 0, 0], [1, 0, 1], [1, 0, 0]], [[0, 0, 0], [0, 0, 1], [1, 0, 1]],
  [[0, 1, 0], [1, 1, 0], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 1]],
  [[0, 0, 0], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
  [[1, 0, 0], [1, 0, 1], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
];

function makeAsciiStl(triangles = cubeTriangles) {
  const facets = triangles.map((vertices) =>
    `facet normal 0 0 1\nouter loop\n${vertices.map(([x, y, z]) => `vertex ${x} ${y} ${z}`).join("\n")}\nendloop\nendfacet`,
  );
  return Buffer.from(`solid test\n${facets.join("\n")}\nendsolid test\n`);
}

test("calculates CI, CVA and CVAI using the provided clinical formulas", () => {
  const indices = calculateCranialIndices(140, 180, 145, 140);
  assert.ok(Math.abs(indices.ci - 77.7777777778) < 1e-8);
  assert.equal(indices.cva, 5);
  assert.ok(Math.abs(indices.cvai - 3.4482758621) < 1e-8);
  assert.throws(() => calculateCranialIndices(140, 0, 145, 140), ScanError);
});

test("parses a closed ASCII STL and calculates reproducible cross-section measurements", () => {
  const scan = parseStl(makeAsciiStl());
  const measured = measureScan(scan);
  assert.equal(scan.triangleCount, 12);
  assert.equal(scan.quality.watertight, true);
  assert.ok(Math.abs(measured.ci - 100) < 1e-8);
  assert.ok(Math.abs(measured.cva) < 1e-8);
  assert.ok(Math.abs(measured.cvai) < 1e-8);
  assert.ok(Math.abs(measured.perimeter - 4) < 1e-6);
  assert.equal(measured.severity, "Ниже порога шкалы");
});

test("applies millimetre scale and manual orientation", () => {
  const scan = parseStl(makeAsciiStl());
  const result = measureScan(scan, {
    scaleMmPerUnit: 2,
    sliceFraction: 0.4,
    diagonalDegrees: 30,
    orientationOffsetDegrees: 12,
  });
  assert.ok(Math.abs(result.perimeter - 8) < 1e-6);
  assert.equal(result.sliceMm, 0.8);
  assert.ok(Math.abs(result.orientationDegrees - result.automaticOrientationDegrees - 12) < 1e-8);
});

test("keeps coordinates centered when the STL is translated and is stable under 3D tilt", () => {
  const translatedText = makeAsciiStl().toString().replace(
    /vertex ([0-1]) ([0-1]) ([0-1])/g,
    (_, x, y, z) => `vertex ${Number(x) + 100} ${Number(y) + 200} ${Number(z) + 300}`,
  );
  const scan = parseStl(Buffer.from(translatedText));
  const result = measureScan(scan, { tiltXDegrees: 20, tiltYDegrees: -12, sliceFraction: 0.5 });
  const centerX = result.contour.reduce((sum, point) => sum + point[0] / result.contour.length, 0);
  const centerY = result.contour.reduce((sum, point) => sum + point[1] / result.contour.length, 0);
  assert.ok(Math.abs(centerX - 100.5) < 1e-4);
  assert.ok(Math.abs(centerY - 200.5) < 1e-4);
  assert.equal(result.tiltXDegrees, 20);
  assert.equal(result.tiltYDegrees, -12);
  assert.deepEqual(result, measureScan(scan, { tiltXDegrees: 20, tiltYDegrees: -12, sliceFraction: 0.5 }));
});


test("handles section planes that pass exactly through mesh vertices and shared edges", () => {
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const triangles = [];
  for (let side = 0; side < 4; side += 1) {
    const next = (side + 1) % 4;
    for (let band = 0; band < 2; band += 1) {
      const low = band * 0.5;
      const high = (band + 1) * 0.5;
      const a = [...corners[side], low];
      const b = [...corners[next], low];
      const c = [...corners[next], high];
      const d = [...corners[side], high];
      triangles.push([a, b, c], [a, c, d]);
    }
  }
  const bottom = corners.map(([x, y]) => [x, y, 0]);
  const top = corners.map(([x, y]) => [x, y, 1]);
  triangles.push([bottom[0], bottom[2], bottom[1]], [bottom[0], bottom[3], bottom[2]]);
  triangles.push([top[0], top[1], top[2]], [top[0], top[2], top[3]]);

  const measured = measureScan(parseStl(makeAsciiStl(triangles)), { sliceFraction: 0.5 });
  assert.ok(Math.abs(measured.perimeter - 4) < 1e-6);
  assert.equal(measured.sectionDiagnostics.closedLoopCount, 1);
  assert.equal(measured.sectionDiagnostics.branchNodeCount, 0);
  assert.equal(measured.sectionDiagnostics.openNodeCount, 0);
});

test("flags incomplete topology and rejects invalid STL and invalid measurement settings", () => {
  const openMesh = parseStl(makeAsciiStl(cubeTriangles.slice(0, 11)));
  assert.equal(openMesh.quality.watertight, false);
  assert.ok(openMesh.quality.boundaryEdges > 0);

  assert.throws(() => parseStl(Buffer.from("not an STL file")), ScanError);
  assert.throws(() => measureScan(parseStl(makeAsciiStl()), { scaleMmPerUnit: -1 }), /Масштаб/);
});
