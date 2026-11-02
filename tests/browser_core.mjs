/* Run with Node.js: node tests/browser_core.mjs. No training, browser or network required. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { decodeModel, predict } from "../docs/inference-core.mjs";
import {
  parseCST,
  surfaces,
  geometryCheck,
  parseCoordinates,
  fitCoordinates,
} from "../docs/geometry-core.mjs";

const root = new URL("../", import.meta.url);
const read = async (path) =>
  JSON.parse(await readFile(new URL(path, root), "utf8"));
const manifest = await read("docs/assets/inference/manifest.json"),
  benchmark = await read("docs/assets/benchmark.json");
const bounds = manifest.bounds;
function close(a, b, tolerance = 1e-9) {
  assert.equal(a.length, b.length);
  a.forEach((v, i) =>
    assert.ok(
      Math.abs(v - b[i]) < tolerance,
      `${v} != ${b[i]}; error=${Math.abs(v - b[i])}`,
    ),
  );
}
function serialize(rows, title = "Coordinate example") {
  return (
    title +
    "\n" +
    rows.map((row) => row.map((v) => v.toPrecision(16)).join(" ")).join("\n")
  );
}
function contour(cst) {
  const points = surfaces(cst, 201);
  return [
    ...points.toReversed().map((p) => [p[0], p[1]]),
    ...points.slice(1).map((p) => [p[0], p[2]]),
  ];
}
const models = {};
for (const family of ["mlp", "kriging"]) {
  const spec = manifest.models[family],
    payload = await readFile(
      new URL("docs/assets/inference/" + spec.file, root),
    );
  assert.equal(createHash("sha256").update(payload).digest("hex"), spec.sha256);
  models[family] = decodeModel(
    spec,
    payload.buffer.slice(
      payload.byteOffset,
      payload.byteOffset + payload.byteLength,
    ),
  );
}
const errors = { mlp: [0, 0, 0], kriging: [0, 0, 0] };
for (const row of benchmark.cases) {
  for (const family of ["mlp", "kriging"]) {
    const actual = predict(models[family], row.cst, family),
      expected = row[family === "mlp" ? "MLP" : "Kriging"];
    close(actual, expected, family === "mlp" ? 1e-9 : 1e-6);
    for (let j = 0; j < 3; j++)
      errors[family][j] = Math.max(
        errors[family][j],
        Math.abs(actual[j] - expected[j]),
      );
    assert.equal(actual[2], actual[0] / actual[1]);
  }
}
const sample = benchmark.cases[0].cst,
  xy = contour(sample),
  three = surfaces(sample, 201);
close(parseCST(sample.join(", ")), sample);
for (const text of [
  "",
  "1 2",
  "NaN ".repeat(20),
  "Infinity ".repeat(20),
  "1e309 ".repeat(20),
  "0x10 ".repeat(20),
])
  assert.throws(() => parseCST(text));
for (const rows of [xy, xy.toReversed(), three]) {
  const fit = fitCoordinates(serialize(rows), bounds, "normalized");
  close(fit.cst, sample, 1e-10);
  assert.ok(fit.maxError < 1e-12);
}
const lednicer =
  "Example\n201. 201.\n" +
  serialize(
    [...three.map((p) => [p[0], p[1]]), ...three.map((p) => [p[0], p[2]])],
    "",
  );
close(fitCoordinates(lednicer, bounds).cst, sample, 1e-10);
const scaled = xy.map(([x, y]) => [1000 * x + 123, 1000 * y - 20]);
const scaledFit = fitCoordinates(serialize(scaled), bounds);
close(scaledFit.cst, sample, 1e-10);
assert.ok(Math.abs(scaledFit.normalization.chord - 1000) < 1e-10);
const angle = (35 * Math.PI) / 180,
  rotate = ([x, y]) => [
    370 * (x * Math.cos(angle) - y * Math.sin(angle)) + 12,
    370 * (x * Math.sin(angle) + y * Math.cos(angle)) - 8,
  ];
const endpoints = [...rotate([0, 0]), ...rotate([1, 0])].join(" "),
  rotated = xy.map(rotate);
const rotatedFit = fitCoordinates(
  serialize(rotated),
  bounds,
  "manual",
  endpoints,
);
close(rotatedFit.cst, sample, 2e-9);
close(
  predict(models.mlp, rotatedFit.cst, "mlp"),
  predict(models.mlp, sample, "mlp"),
  1e-9,
);
assert.throws(
  () => fitCoordinates(serialize(rotated), bounds, "aligned"),
  /horizontal chord/,
);
assert.throws(() => fitCoordinates(serialize(scaled), bounds, "normalized"));
const shuffled = xy.map((p) => [...p]);
[shuffled[20], shuffled[25]] = [shuffled[25], shuffled[20]];
assert.throws(() => fitCoordinates(serialize(shuffled), bounds), /turn back/);
assert.throws(
  () => fitCoordinates(serialize(three.map(([x, u, l]) => [x, l, u])), bounds),
  /signed ordinates/,
);
assert.throws(
  () =>
    fitCoordinates(
      serialize(three.map(([x, u, l]) => [x, u + 0.001 * x, l - 0.001 * x])),
      bounds,
    ),
  /edge is open/,
);
for (const bad of ["", "nan nan\n".repeat(100), "0 0\nbad", "1 2\n3 4 5"])
  assert.throws(() => parseCoordinates(bad));
assert.throws(
  () => parseCoordinates(serialize(three.slice(30, -30)), "normalized"),
  /endpoint/,
);
const outside = [...sample];
outside[0] = bounds.upper[0] + 0.001;
assert.ok(geometryCheck(outside, bounds).outside.includes("U0"));
assert.throws(
  () => geometryCheck([...sample.slice(10), ...sample.slice(0, 10)], bounds),
  /cross/,
);
// Independent non-CST geometry can fit well while remaining outside the model's design box.
const naca = three.map(([x]) => {
  const y =
    0.6 *
    (0.2969 * Math.sqrt(x) -
      0.126 * x -
      0.3516 * x * x +
      0.2843 * x ** 3 -
      0.1036 * x ** 4);
  return [x, y, -y];
});
assert.ok(fitCoordinates(serialize(naca), bounds).geometry.outside.length > 0);
// A distorted outline cannot silently pass the geometric fit gate.
const noisy = three.map(([x, u, l]) => [
  x,
  u + 0.04 * x * (1 - x) * Math.sin(40 * Math.PI * x),
  l,
]);
assert.throws(() => fitCoordinates(serialize(noisy), bounds), /too inaccurate/);
console.log(
  JSON.stringify(
    {
      status: "passed",
      test_cases: benchmark.cases.length,
      max_absolute_errors: errors,
      checks: [
        "model checksums",
        "MLP and Kriging parity",
        "CST parser",
        "three-column, reversed contour, Lednicer",
        "translation, units and explicit rotation",
        "bad order, open edges, missing endpoints, nonfinite input",
        "out-of-domain geometry and poor fit rejection",
      ],
    },
    null,
    2,
  ),
);
