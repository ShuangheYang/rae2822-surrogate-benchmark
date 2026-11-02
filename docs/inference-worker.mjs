import { decodeModel, predict } from "./inference-core.mjs";
import { parseCST, geometryCheck, fitCoordinates } from "./geometry-core.mjs";

let manifestPromise;
const models = new Map();
async function readManifest() {
  if (!manifestPromise)
    manifestPromise = fetch(
      new URL("assets/inference/manifest.json", import.meta.url),
      { cache: "no-store", credentials: "omit" },
    )
      .then((response) => {
        if (!response.ok)
          throw new Error("Model manifest could not be loaded. Please retry.");
        return response.json();
      })
      .catch((error) => {
        manifestPromise = null;
        throw error;
      });
  return manifestPromise;
}
async function loadModel(family, manifest) {
  if (!models.has(family)) {
    models.set(
      family,
      (async () => {
        const spec = manifest.models[family],
          url = new URL(`assets/inference/${spec.file}`, import.meta.url);
        const response = await fetch(url, { credentials: "omit" });
        if (!response.ok)
          throw new Error(`${family} model download failed. Please retry.`);
        const bytes = await response.arrayBuffer();
        const digest = Array.from(
          new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
          (v) => v.toString(16).padStart(2, "0"),
        ).join("");
        if (digest !== spec.sha256)
          throw new Error(
            "Model checksum mismatch. Reload the page before predicting.",
          );
        return decodeModel(spec, bytes);
      })().catch((error) => {
        models.delete(family);
        throw error;
      }),
    );
  }
  return models.get(family);
}

self.onmessage = async ({ data: request }) => {
  const {
    id,
    mode,
    text,
    coordinateMode,
    chordPoints,
    compareGP,
    allowOutside,
  } = request;
  try {
    self.postMessage({
      id,
      status:
        mode === "coordinates"
          ? "Normalizing and fitting coordinates…"
          : "Checking CST geometry…",
    });
    const manifest = await readManifest();
    let analysis;
    if (mode === "cst") {
      const cst = parseCST(text);
      analysis = { cst, geometry: geometryCheck(cst, manifest.bounds) };
    } else if (mode === "coordinates")
      analysis = fitCoordinates(
        text,
        manifest.bounds,
        coordinateMode,
        chordPoints,
      );
    else throw new Error("Unknown input mode.");
    if (analysis.geometry.outside.length && !allowOutside) {
      self.postMessage({
        id,
        analysis,
        blocked: true,
        result: null,
        message: `Outside the trained CST box: ${analysis.geometry.outside.join(", ")}. Inspect the geometry and enable out-of-domain exploration to predict.`,
      });
      return;
    }
    const result = {},
      references = {};
    for (const family of compareGP ? ["mlp", "kriging"] : ["mlp"]) {
      self.postMessage({
        id,
        status: `Loading ${family === "mlp" ? "MLP" : "Kriging"} and calculating…`,
      });
      const model = await loadModel(family, manifest);
      result[family] = predict(model, analysis.cst, family);
      const spec = manifest.models[family];
      references[family] = {
        source_url: spec.source_url,
        source_npz_sha256: spec.source_npz_sha256,
        browser_sha256: spec.sha256,
      };
    }
    self.postMessage({
      id,
      analysis,
      result,
      models: references,
      message: analysis.geometry.outside.length
        ? "Prediction complete · out-of-domain exploration. Accuracy is unvalidated."
        : "Prediction complete · CST coefficients are inside the studied parameter box.",
    });
  } catch (error) {
    self.postMessage({ id, error: error.message || String(error) });
  }
};
