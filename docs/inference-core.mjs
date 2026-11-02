/* Numerical forward passes for the frozen released checkpoints. All arithmetic is float64. */
export function decodeModel(spec, buffer) {
  if (buffer.byteLength !== spec.bytes)
    throw new Error("Incomplete model download. Reload and try again.");
  const view = new DataView(buffer),
    values = new Float64Array(buffer.byteLength / 8);
  for (let i = 0; i < values.length; i++)
    values[i] = view.getFloat64(8 * i, true);
  const arrays = {};
  for (const [name, layout] of Object.entries(spec.arrays)) {
    arrays[name] = values.subarray(
      layout.offset,
      layout.offset + layout.length,
    );
  }
  return { arrays, spec };
}

function dense(input, weights, bias, rows) {
  const output = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let value = bias ? bias[i] : 0;
    for (let j = 0; j < input.length; j++)
      value += weights[i * input.length + j] * input[j];
    output[i] = value;
  }
  return output;
}

export function predict(model, cst, family) {
  if (cst.length !== 20 || !cst.every(Number.isFinite))
    throw new Error("Expected 20 finite CST coefficients.");
  const { arrays: a, spec } = model,
    definition = spec.definition;
  const z = Float64Array.from(cst, (v, i) => (v - a.x_mean[i]) / a.x_scale[i]);
  let output;
  if (family === "mlp") {
    let hidden = z;
    definition.layers.forEach((layer, index) => {
      hidden = dense(
        hidden,
        a[layer.weight],
        a[layer.bias],
        a[layer.bias].length,
      );
      if (index < definition.layers.length - 1) hidden = hidden.map(Math.tanh);
    });
    if (definition.skip) {
      const linear = dense(z, a[definition.skip], null, 2);
      hidden[0] += linear[0];
      hidden[1] += linear[1];
    }
    output = Array.from(hidden, (v, i) => v * a.y_scale[i] + a.y_mean[i]);
    if (definition.log_cd) output[1] = Math.exp(output[1]);
  } else if (family === "kriging") {
    const sums = [0, 0],
      alpha = [a.alpha_CL, a.alpha_CD];
    let qnorm = 0;
    for (const v of z) qnorm += v * v;
    for (let i = 0; i < definition.training_points; i++) {
      let dot = 0,
        norm = 0;
      for (let j = 0; j < 20; j++) {
        const v = a.x_train[i * 20 + j];
        dot += z[j] * v;
        norm += v * v;
      }
      const distance = Math.sqrt(Math.max(-2 * dot + qnorm + norm, 0));
      for (let j = 0; j < 2; j++) {
        const p = definition.parameters[j],
          u = (Math.sqrt(5) * distance) / p.length_scale;
        sums[j] +=
          p.signal * (1 + u + (u * u) / 3) * Math.exp(-u) * alpha[j][i];
      }
    }
    output = sums.map((v, i) => v * a.y_scale[i] + a.y_mean[i]);
  } else throw new Error("Unknown model family.");
  if (!output.every(Number.isFinite) || output[1] <= 0)
    throw new Error(
      "Model returned a nonfinite value or nonpositive drag. Outputs were not clipped.",
    );
  const ratio = output[0] / output[1];
  if (!Number.isFinite(ratio))
    throw new Error("Model returned a nonfinite lift-to-drag ratio.");
  return [...output, ratio];
}
