/* Chord normalization and degree-9 CST least-squares fitting for closed airfoil surfaces. */
const BINOMIAL = [1, 9, 36, 84, 126, 126, 84, 36, 9, 1];
const MAX_TEXT = 1000000,
  MIN_POINTS = 30;
export const FIT_LIMITS = { rmse: 5e-4, maxError: 2e-3 };
const finite = (values) => values.every(Number.isFinite);
const numeric = (token) =>
  /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$/.test(token);
const number = (token) => Number(token.replace(/[dD]/g, "e"));

export function parseCST(text) {
  const fields = String(text)
    .trim()
    .split(/[\s,;\[\]]+/)
    .filter(Boolean);
  if (fields.length !== 20 || !fields.every(numeric))
    throw new Error(
      `Enter exactly 20 finite numbers in U0–U9, L0–L9 order (received ${fields.length}).`,
    );
  const result = fields.map(number);
  if (!finite(result)) throw new Error("CST coefficients must be finite.");
  return result;
}

export function basis(x) {
  return BINOMIAL.map(
    (v, i) => Math.sqrt(x) * (1 - x) * v * x ** i * (1 - x) ** (9 - i),
  );
}
function ordinate(coefficients, x, offset) {
  return basis(x).reduce((sum, v, i) => sum + v * coefficients[offset + i], 0);
}
export function surfaces(cst, count = 501) {
  return Array.from({ length: count }, (_, i) => {
    const x = (1 - Math.cos((Math.PI * i) / (count - 1))) / 2;
    return [x, ordinate(cst, x, 0), ordinate(cst, x, 10)];
  });
}
export function geometryCheck(cst, bounds) {
  if (cst.length !== 20 || !finite(cst))
    throw new Error("Expected 20 finite CST coefficients.");
  let maxThickness = 0,
    maxCamber = 0,
    thicknessX = 0;
  for (const [x, u, l] of surfaces(cst)) {
    if (!finite([x, u, l]))
      throw new Error("Nonfinite reconstructed geometry.");
    if (x > 0 && x < 1 && u <= l)
      throw new Error(
        "Upper and lower surfaces cross or have nonpositive interior thickness.",
      );
    if (u - l > maxThickness) {
      maxThickness = u - l;
      thicknessX = x;
    }
    maxCamber = Math.max(maxCamber, Math.abs((u + l) / 2));
  }
  if (maxThickness <= 0.04 || maxThickness >= 0.25 || maxCamber > 0.1)
    throw new Error(
      "Implausible thickness or camber. Check coefficient order, units and surface signs.",
    );
  const outside = cst
    .map((v, i) =>
      v < bounds.lower[i] - 1e-10 || v > bounds.upper[i] + 1e-10
        ? bounds.order[i]
        : null,
    )
    .filter(Boolean);
  return { maxThickness, maxCamber, thicknessX, outside };
}

function readRows(text) {
  if (typeof text !== "string" || !text.trim() || text.length > MAX_TEXT)
    throw new Error("Provide coordinate text up to 1 MB.");
  const rows = [];
  for (const [index, raw] of text
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .entries()) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || line.startsWith("!")) continue;
    const tokens = line.split(/[\s,;]+/);
    if (!tokens.every(numeric)) {
      if (
        !rows.length &&
        /[a-z]/i.test(line) &&
        !/(?:\bnan\b|\binf(?:inity)?\b)/i.test(line)
      )
        continue;
      throw new Error(
        `Non-numeric coordinate row ${index + 1}. Only leading titles/headers and # comments may be skipped.`,
      );
    }
    if (![2, 3].includes(tokens.length))
      throw new Error(
        `Row ${index + 1}: use two columns x y or three columns x y_upper y_lower.`,
      );
    const values = tokens.map(number);
    if (!finite(values))
      throw new Error("Coordinates cannot contain NaN or Infinity.");
    rows.push(values);
  }
  if (
    !rows.length ||
    rows.length > 20000 ||
    rows.some((r) => r.length !== rows[0].length)
  )
    throw new Error(
      "Coordinate rows must have consistent columns, with at most 20,000 rows.",
    );
  let counts = null;
  if (
    rows[0].length === 2 &&
    rows[0].every((v) => Number.isInteger(v) && v >= MIN_POINTS) &&
    rows[0][0] + rows[0][1] === rows.length - 1
  )
    counts = rows.shift();
  return { rows, counts };
}

function normalization(points, mode, chordPoints) {
  let le, te;
  if (mode === "normalized") {
    le = [0, 0];
    te = [1, 0];
  } else if (mode === "aligned") {
    let min = Infinity,
      max = -Infinity;
    for (const p of points) {
      min = Math.min(min, p[0]);
      max = Math.max(max, p[0]);
    }
    if (max <= min) throw new Error("Chord length must be positive.");
    const at = (x) =>
      points.filter((p) => Math.abs(p[0] - x) <= (max - min) * 1e-10);
    const meanY = (list) => list.reduce((s, p) => s + p[1], 0) / list.length;
    le = [min, meanY(at(min))];
    te = [max, meanY(at(max))];
    if (Math.abs(te[1] - le[1]) / (max - min) > 1e-5)
      throw new Error(
        "Automatic normalization requires a horizontal chord, leading edge on the left. For rotated coordinates, select explicit LE/TE and enter both endpoints.",
      );
  } else if (mode === "manual") {
    const tokens = String(chordPoints)
      .trim()
      .split(/[\s,;]+/);
    if (tokens.length !== 4 || !tokens.every(numeric))
      throw new Error(
        "Enter four chord endpoint numbers: x_LE y_LE x_TE y_TE.",
      );
    const v = tokens.map(number);
    if (!finite(v)) throw new Error("Chord endpoints must be finite.");
    le = v.slice(0, 2);
    te = v.slice(2);
  } else throw new Error("Unknown normalization mode.");
  const dx = te[0] - le[0],
    dy = te[1] - le[1],
    chord = Math.hypot(dx, dy);
  if (!Number.isFinite(chord) || chord <= 0)
    throw new Error("Leading and trailing edge points must be distinct.");
  const transform = (p) => [
    ((p[0] - le[0]) * dx + (p[1] - le[1]) * dy) / chord ** 2,
    (-(p[0] - le[0]) * dy + (p[1] - le[1]) * dx) / chord ** 2,
  ];
  return {
    transform,
    meta: {
      mode,
      leadingEdge: le,
      trailingEdge: te,
      chord,
      rotationDegrees: (Math.atan2(dy, dx) * 180) / Math.PI,
    },
  };
}

function interpolate(branch, x) {
  let lo = 0,
    hi = branch.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (branch[mid][0] <= x) lo = mid;
    else hi = mid;
  }
  const a = branch[lo],
    b = branch[hi],
    t = (x - a[0]) / (b[0] - a[0]);
  return a[1] + t * (b[1] - a[1]);
}

export function parseCoordinates(text, mode = "aligned", chordPoints = "") {
  const { rows, counts } = readRows(text),
    triple = rows[0].length === 3;
  if (triple && rows.some((r) => r[1] < r[2] - 1e-12))
    throw new Error(
      "Three-column order is x y_upper y_lower. Use signed ordinates, not unsigned lower thickness.",
    );
  const points = triple
    ? rows.flatMap((r) => [
        [r[0], r[1]],
        [r[0], r[2]],
      ])
    : rows;
  const { transform, meta } = normalization(points, mode, chordPoints);
  let branches;
  if (triple)
    branches = [
      rows.map((r) => transform([r[0], r[1]])),
      rows.map((r) => transform([r[0], r[2]])),
    ];
  else {
    const xy = rows.map(transform);
    if (counts) branches = [xy.slice(0, counts[0]), xy.slice(counts[0])];
    else {
      let split = xy.reduce((best, p, i) => (p[0] < xy[best][0] ? i : best), 0);
      if (split === 0 || split === xy.length - 1)
        split = xy.reduce((best, p, i) => (p[0] > xy[best][0] ? i : best), 0);
      branches = [xy.slice(0, split + 1), xy.slice(split)];
    }
  }
  branches = branches.map((branch) => {
    if (branch.length < MIN_POINTS)
      throw new Error(
        "Each surface needs at least 30 points. Use an ordered contour or a Lednicer point-count header.",
      );
    const differences = branch.slice(1).map((p, i) => p[0] - branch[i][0]);
    if (differences.some((v) => v > 1e-8) && differences.some((v) => v < -1e-8))
      throw new Error(
        "Surface x coordinates turn back. Unordered point clouds are not supported.",
      );
    const sorted = [...branch].sort((a, b) => a[0] - b[0]),
      clean = [];
    for (const p of sorted) {
      const previous = clean.at(-1);
      if (previous && Math.abs(p[0] - previous[0]) < 1e-12) {
        if (Math.abs(p[1] - previous[1]) > 1e-5)
          throw new Error("A surface has conflicting ordinates at the same x.");
      } else clean.push(p);
    }
    if (clean.length < MIN_POINTS)
      throw new Error(
        "Each surface needs at least 30 distinct chordwise positions.",
      );
    const first = clean[0][0],
      last = clean.at(-1)[0];
    if (first < -1e-6 || last > 1 + 1e-6 || first > 0.001 || last < 0.999)
      throw new Error(
        "Each surface must cover the leading and trailing edges (within 0.001 chord). Check endpoint normalization.",
      );
    return clean.map(([x, y]) => [Math.min(1, Math.max(0, x)), y]);
  });
  const grid = Array.from({ length: 999 }, (_, i) => 0.01 + (0.98 * i) / 998);
  branches.sort((a, b) =>
    grid.reduce((sum, x) => sum + interpolate(b, x) - interpolate(a, x), 0),
  );
  if (
    grid.some((x) => interpolate(branches[0], x) <= interpolate(branches[1], x))
  )
    throw new Error(
      "Input upper and lower surfaces cross or have nonpositive interior thickness.",
    );
  for (const side of [0, -1]) {
    const ends = branches.map((b) => b.at(side)),
      endpoint = side === 0 ? 0 : 1;
    if (
      ends.every((p) => Math.abs(p[0] - endpoint) < 1e-6) &&
      Math.abs(ends[0][1] - ends[1][1]) > 1e-5
    )
      throw new Error(
        `${side === 0 ? "Leading" : "Trailing"} edge is open. This CST definition has closed endpoints and no finite trailing-edge term.`,
      );
  }
  return {
    branches,
    normalization: {
      ...meta,
      format: triple
        ? "x y_upper y_lower"
        : counts
          ? "Lednicer contour"
          : "Ordered contour",
      points: branches.map((b) => b.length),
    },
  };
}

export function leastSquares(matrix, target) {
  // Column-pivoted Householder QR; avoids squaring the CST basis condition number.
  const a = matrix.map((row) => [...row]),
    b = [...target],
    n = a[0].length,
    m = a.length;
  const permutation = Array.from({ length: n }, (_, i) => i);
  let initialNorm = 0;
  for (let k = 0; k < n; k++) {
    let best = k,
      largest = -1;
    for (let j = k; j < n; j++) {
      let norm = 0;
      for (let i = k; i < m; i++) norm += a[i][j] ** 2;
      if (norm > largest) {
        largest = norm;
        best = j;
      }
    }
    if (k === 0) initialNorm = Math.sqrt(largest);
    if (Math.sqrt(largest) < initialNorm * 1e-12)
      throw new Error(
        "CST fit is rank deficient. Supply well-spaced coordinates across both surfaces.",
      );
    if (best !== k) {
      for (const row of a) [row[k], row[best]] = [row[best], row[k]];
      [permutation[k], permutation[best]] = [permutation[best], permutation[k]];
    }
    const v = a.slice(k).map((row) => row[k]),
      norm = Math.sqrt(v.reduce((s, t) => s + t * t, 0));
    v[0] += (v[0] >= 0 ? 1 : -1) * norm;
    const vnorm = Math.sqrt(v.reduce((s, t) => s + t * t, 0));
    for (let i = 0; i < v.length; i++) v[i] /= vnorm;
    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += v[i - k] * a[i][j];
      for (let i = k; i < m; i++) a[i][j] -= 2 * v[i - k] * dot;
    }
    let dot = 0;
    for (let i = k; i < m; i++) dot += v[i - k] * b[i];
    for (let i = k; i < m; i++) b[i] -= 2 * v[i - k] * dot;
  }
  const z = new Array(n).fill(0),
    result = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let value = b[i];
    for (let j = i + 1; j < n; j++) value -= a[i][j] * z[j];
    z[i] = value / a[i][i];
  }
  for (let i = 0; i < n; i++) result[permutation[i]] = z[i];
  if (!finite(result)) throw new Error("Nonfinite CST fit.");
  return result;
}

export function fitCoordinates(
  text,
  bounds,
  mode = "aligned",
  chordPoints = "",
) {
  const packet = parseCoordinates(text, mode, chordPoints),
    cst = [],
    rmse = [];
  let maxError = 0;
  for (const branch of packet.branches) {
    const a = branch.map((p) => basis(p[0])),
      coefficients = leastSquares(
        a,
        branch.map((p) => p[1]),
      );
    cst.push(...coefficients);
    let squared = 0;
    branch.forEach((p, i) => {
      const error = a[i].reduce((s, v, j) => s + v * coefficients[j], 0) - p[1];
      squared += error * error;
      maxError = Math.max(maxError, Math.abs(error));
    });
    rmse.push(Math.sqrt(squared / branch.length));
  }
  if (Math.max(...rmse) > FIT_LIMITS.rmse || maxError > FIT_LIMITS.maxError)
    throw new Error(
      `CST fit is too inaccurate: maximum surface RMSE ${Math.max(...rmse).toExponential(3)} c; maximum deviation ${maxError.toExponential(3)} c. Limits are 0.0005 c and 0.002 c.`,
    );
  return {
    ...packet,
    cst,
    rmse,
    maxError,
    geometry: geometryCheck(cst, bounds),
  };
}
