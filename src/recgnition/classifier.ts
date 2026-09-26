/**
 * Letter classifier: landmarks in, letter + confidence out.
 *
 * Supported vocabulary is whatever the Python training set actually contains
 * (24 static fingerspelling handshapes — J and Z are excluded because they
 * require motion and were never collected). Nothing outside `model.labels`
 * can ever be produced.
 */
import modelJson from "./model/letter-model.json";
import { predictProbabilities, type ExportedModel } from "./mlp";
import { normalizeLandmarks, type Landmark } from "./normalize";

const model = modelJson as unknown as ExportedModel;

export const SUPPORTED_LETTERS: string[] = model.labels.map((l) => l.toUpperCase());

export interface Prediction {
  letter: string;
  confidence: number;
  /** Runner-up, useful for showing how close a call was. */
  alternative: { letter: string; confidence: number } | null;
  inferenceMs: number;
}

export function classifyLandmarks(landmarks: Landmark[]): Prediction | null {
  if (!landmarks || landmarks.length !== 21) return null;

  const started = performance.now();
  const features = normalizeLandmarks(landmarks);
  if (features.length !== model.featureCount) return null;

  const probabilities = predictProbabilities(model, features);

  let best = 0;
  let second = -1;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[best]) best = i;
  }
  for (let i = 0; i < probabilities.length; i++) {
    if (i === best) continue;
    if (second === -1 || probabilities[i] > probabilities[second]) second = i;
  }

  return {
    letter: model.labels[best].toUpperCase(),
    confidence: probabilities[best],
    alternative:
      second >= 0
        ? { letter: model.labels[second].toUpperCase(), confidence: probabilities[second] }
        : null,
    inferenceMs: performance.now() - started,
  };
}
