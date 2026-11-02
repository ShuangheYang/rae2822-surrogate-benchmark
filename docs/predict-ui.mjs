import { surfaces } from "./geometry-core.mjs";

const $ = (id) => document.getElementById(id),
  NS = "http://www.w3.org/2000/svg";
let revision = 0,
  active = 0,
  worker,
  lastPacket = null;
let manifestPromise;
const manifest = () =>
  manifestPromise ||
  (manifestPromise = fetch("assets/inference/manifest.json", {
    cache: "no-store",
  })
    .then((r) => {
      if (!r.ok)
        throw new Error("Cannot load examples. Retry after reloading.");
      return r.json();
    })
    .catch((e) => {
      manifestPromise = null;
      throw e;
    }));
const formatCST = (cst) =>
  [cst.slice(0, 10), cst.slice(10)]
    .map((a) => a.map((v) => v.toPrecision(16)).join(", "))
    .join("\n");
function svgNode(tag, attributes, text = "") {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attributes)) el.setAttribute(k, v);
  if (text) el.textContent = text;
  return el;
}
function invalidate(
  message = "Input changed. Run prediction to update the results.",
) {
  revision++;
  active = 0;
  lastPacket = null;
  $("live-output").hidden = true;
  $("predict-submit").disabled = false;
  $("live-status").textContent = message;
  $("live-status").className = "note";
}
function mode() {
  return document.querySelector('input[name="input-mode"]:checked').value;
}
function selectMode(value) {
  document.querySelector(`input[name="input-mode"][value="${value}"]`).checked =
    true;
  updateMode();
}
function updateMode() {
  const coordinate = mode() === "coordinates";
  $("coordinate-options").hidden = !coordinate;
  $("input-label").textContent = coordinate
    ? "Airfoil surface coordinates"
    : "20 CST coefficients · U0–U9, L0–L9";
  $("custom-input").placeholder = coordinate
    ? "Airfoil name (optional)\n1.000000 0.000000\n0.999013 0.000219\n…"
    : "Paste 20 numbers separated by commas, spaces or newlines.";
  $("predict-submit").textContent = coordinate
    ? "Fit CST & predict"
    : "Predict aerodynamic coefficients";
  invalidate("Enter your geometry, or load an example below.");
}
function displayAnalysis(analysis) {
  $("live-output").hidden = false;
  $("fitted-cst").textContent = formatCST(analysis.cst);
  const g = analysis.geometry;
  $("live-geometry-info").textContent =
    `Maximum thickness: ${g.maxThickness.toFixed(6)} c at x/c=${g.thicknessX.toFixed(4)} · Maximum absolute camber: ${g.maxCamber.toFixed(6)} c`;
  $("fit-info").hidden = !analysis.rmse;
  if (analysis.rmse) {
    const n = analysis.normalization;
    $("fit-info").textContent =
      `${n.format}: ${n.points[0]} upper / ${n.points[1]} lower points. Input chord ${n.chord.toPrecision(7)}; chord rotation ${n.rotationDegrees.toFixed(3)}°. Upper RMSE ${analysis.rmse[0].toExponential(3)} c; lower RMSE ${analysis.rmse[1].toExponential(3)} c; maximum deviation ${analysis.maxError.toExponential(3)} c. Prediction uses this fitted CST geometry.`;
  }
  const svg = $("live-airfoil");
  svg.replaceChildren();
  svg.append(
    svgNode("line", {
      x1: 60,
      y1: 158,
      x2: 945,
      y2: 158,
      stroke: "#d8d0e0",
      "stroke-dasharray": "4 5",
    }),
  );
  for (let i = 0; i <= 5; i++)
    svg.append(
      svgNode(
        "text",
        {
          x: 70 + (860 * i) / 5,
          y: 268,
          "text-anchor": "middle",
          "font-size": 15,
          fill: "#70697a",
        },
        (i / 5).toFixed(1),
      ),
    );
  svg.append(
    svgNode(
      "text",
      {
        x: 500,
        y: 292,
        "text-anchor": "middle",
        "font-size": 15,
        fill: "#70697a",
      },
      "x / c",
    ),
  );
  if (analysis.branches)
    for (const branch of analysis.branches) {
      const stride = Math.max(1, Math.ceil(branch.length / 250));
      for (let i = 0; i < branch.length; i += stride) {
        const [x, y] = branch[i];
        svg.append(
          svgNode("circle", {
            cx: 70 + 860 * x,
            cy: 158 - 860 * y,
            r: 2.2,
            fill: "#428b95",
            opacity: 0.65,
          }),
        );
      }
    }
  const points = surfaces(analysis.cst, 301);
  for (const side of [1, 2])
    svg.append(
      svgNode("path", {
        d: points
          .map(
            (r, i) =>
              `${i ? "L" : "M"}${70 + 860 * r[0]},${158 - 860 * r[side]}`,
          )
          .join(" "),
        fill: "none",
        stroke: "#73508e",
        "stroke-width": 2.5,
      }),
    );
}
function displayResult(packet) {
  displayAnalysis(packet.analysis);
  const body = $("live-predictions");
  body.replaceChildren();
  $("live-result-table").hidden = !packet.result;
  $("prediction-disclaimer").hidden = !packet.result;
  $("save-prediction").disabled = !packet.result;
  if (packet.result) {
    for (const [j, label] of ["CL", "CD", "CL / CD"].entries()) {
      const row = document.createElement("tr");
      for (const value of [
        label,
        packet.result.mlp[j].toPrecision(9),
        packet.result.kriging
          ? packet.result.kriging[j].toPrecision(9)
          : "Not requested",
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
  }
  $("live-status").textContent = packet.message;
  $("live-status").className =
    packet.blocked || packet.analysis.geometry.outside.length
      ? "note caution"
      : "note success";
  lastPacket = packet;
}
function makeWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("./inference-worker.mjs", import.meta.url), {
    type: "module",
  });
  worker.onmessage = ({ data: packet }) => {
    if (packet.id !== active || packet.id !== revision) return;
    if (packet.status) {
      $("live-status").textContent = packet.status;
      return;
    }
    $("predict-submit").disabled = false;
    if (packet.error) {
      $("live-status").textContent = packet.error;
      $("live-status").className = "note caution";
    } else displayResult(packet);
  };
  worker.onerror = () => {
    worker.terminate();
    worker = null;
    active = 0;
    $("predict-submit").disabled = false;
    $("live-status").textContent =
      "Browser prediction could not start. Use the HTTPS website in a current browser, then retry.";
    $("live-status").className = "note caution";
  };
  return worker;
}

$("live-form").addEventListener("submit", (event) => {
  event.preventDefault();
  invalidate("Preparing browser inference…");
  active = revision;
  $("predict-submit").disabled = true;
  try {
    makeWorker().postMessage({
      id: active,
      mode: mode(),
      text: $("custom-input").value,
      coordinateMode: $("coordinate-mode").value,
      chordPoints: $("chord-points").value,
      compareGP: $("compare-kriging").checked,
      allowOutside: $("allow-outside").checked,
    });
  } catch (error) {
    $("predict-submit").disabled = false;
    $("live-status").textContent = error.message;
  }
});
document
  .querySelectorAll('input[name="input-mode"]')
  .forEach((el) => el.addEventListener("change", updateMode));
for (const id of ["custom-input", "chord-points"])
  $(id).addEventListener("input", () => invalidate());
for (const id of ["compare-kriging", "allow-outside"])
  $(id).addEventListener("change", () => invalidate());
$("coordinate-mode").addEventListener("change", () => {
  $("manual-chord").hidden = $("coordinate-mode").value !== "manual";
  invalidate();
});
$("coordinate-file").addEventListener("change", async () => {
  const file = $("coordinate-file").files[0];
  invalidate("Reading the coordinate file locally…");
  const ticket = revision;
  if (!file) return;
  try {
    if (file.size > 1000000)
      throw new Error("Choose a coordinate text file smaller than 1 MB.");
    const text = await file.text();
    if (ticket !== revision) return;
    selectMode("coordinates");
    $("custom-input").value = text;
    $("live-status").textContent =
      `Loaded ${file.name}. Select normalization and run Fit CST & predict.`;
  } catch (error) {
    if (ticket === revision) {
      $("live-status").textContent = error.message;
      $("live-status").className = "note caution";
    }
  }
});
for (const kind of ["cst", "coordinates"])
  $(`example-${kind}`).addEventListener("click", async () => {
    invalidate("Loading the baseline example…");
    const ticket = revision;
    try {
      const meta = await manifest();
      if (ticket !== revision) return;
      selectMode(kind);
      $("custom-input").value =
        kind === "cst"
          ? formatCST(meta.bounds.baseline)
          : "RAE2822 CST baseline\n# x y_upper y_lower\n" +
            surfaces(meta.bounds.baseline, 201)
              .map((r) => r.map((v) => v.toPrecision(16)).join(" "))
              .join("\n");
      $("coordinate-mode").value = "aligned";
      $("manual-chord").hidden = true;
      $("chord-points").value = "";
      $("live-status").textContent =
        "Baseline example loaded. Run prediction to calculate new outputs.";
    } catch (error) {
      $("live-status").textContent = error.message;
    }
  });
$("use-explorer").addEventListener("click", () =>
  window.dispatchEvent(new Event("request-cst-example")),
);
window.addEventListener("cst-example", (event) => {
  selectMode("cst");
  $("custom-input").value = formatCST(event.detail.cst);
  $("live-status").textContent =
    `Copied LHS_2000:${event.detail.id}. Run prediction to recompute with the model weights.`;
});
$("save-prediction").addEventListener("click", () => {
  if (!lastPacket?.result) return;
  const payload = {
    schema: 1,
    created_at: new Date().toISOString(),
    condition: { Mach: 0.73, Reynolds: 6500000, AoA_deg: 2.79 },
    outputs: ["CL", "CD", "CL/CD"],
    cst: lastPacket.analysis.cst,
    geometry: lastPacket.analysis.geometry,
    fit: lastPacket.analysis.rmse
      ? {
          rmse: lastPacket.analysis.rmse,
          maxError: lastPacket.analysis.maxError,
          normalization: lastPacket.analysis.normalization,
        }
      : null,
    prediction: lastPacket.result,
    models: lastPacket.models,
    model_manifest: new URL("assets/inference/manifest.json", location.href)
      .href,
    interpretation:
      "Surrogate prediction at the specified fixed flow condition. Coordinate fit error describes geometry only.",
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "airfoil_prediction.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
updateMode();
