/**
 * Letter classifier: landmarks in, letter + confidence out.
 *
 * Supported vocabulary is whatever the Python training set actually contains
 * (24 static fingerspelling handshapes — J and Z are excluded because they
 * require motion and were never collected). Nothing outside `model.labels`
 * can ever be produced.
 */
import modelJson from "./model/letter-model.json";
import { predictProbabilities, type ExportedModel } from "./mlp.ts";
import { normalizeLandmarks, type Landmark } from "./normalize.ts";

const model = modelJson as unknown as ExportedModel;

export const SUPPORTED_LETTERS: string[] = model.labels.map((label: string) => label.toUpperCase());

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
  if (probabilities.length === 0 || model.labels.length === 0) return null;

  let best = 0;
  let second = -1;
  for (let i = 1; i < probabilities.length; i++) {
    const probability = probabilities[i];
    const bestProbability = probabilities[best];
    if (probability !== undefined && bestProbability !== undefined && probability > bestProbability) {
      best = i;
    }
  }
  for (let i = 0; i < probabilities.length; i++) {
    if (i === best) continue;
    const probability = probabilities[i];
    const secondProbability = second >= 0 ? probabilities[second] : undefined;
    if (probability !== undefined && (second === -1 || secondProbability === undefined || probability > secondProbability)) {
      second = i;
    }
  }

  const letter = model.labels[best];
  const confidence = probabilities[best];
  if (letter === undefined || confidence === undefined) return null;
  const alternativeLetter = second >= 0 ? model.labels[second] : undefined;
  const alternativeConfidence = second >= 0 ? probabilities[second] : undefined;

  return {
    letter: letter.toUpperCase(),
    confidence,
    alternative:
      alternativeLetter !== undefined && alternativeConfidence !== undefined
        ? { letter: alternativeLetter.toUpperCase(), confidence: alternativeConfidence }
        : null,
    inferenceMs: performance.now() - started,
  };
}
