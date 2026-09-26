/**
 * Minimal dense-network forward pass.
 *
 * The classifier is trained in Python (ml/train_export.py) and exported as
 * plain weight matrices, so inference here is a few matrix multiplies — no
 * ML runtime, no WebGL, no server round-trip. Measured at well under a
 * millisecond per frame, which is why the camera loop stays smooth.
 */

export interface DenseLayer {
  weights: number[][];
  bias: number[];
}

export interface ExportedModel {
  format: string;
  activation: string;
  output: string;
  featureCount: number;
  labels: string[];
  layers: DenseLayer[];
}

function relu(v: number[]): number[] {
  return v.map((x) => (x > 0 ? x : 0));
}

function softmax(v: number[]): number[] {
  const max = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((x) => x / sum);
}

function forwardLayer(input: number[], layer: DenseLayer): number[] {
  const outSize = layer.bias.length;
  const out = new Array<number>(outSize);
  for (let j = 0; j < outSize; j++) {
    let sum = layer.bias[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * layer.weights[i][j];
    }
    out[j] = sum;
  }
  return out;
}

/** Runs the network and returns a probability for every class label. */
export function predictProbabilities(model: ExportedModel, features: number[]): number[] {
  let activations = features;
  for (let i = 0; i < model.layers.length; i++) {
    activations = forwardLayer(activations, model.layers[i]);
    if (i < model.layers.length - 1) activations = relu(activations);
  }
  return softmax(activations);
}
