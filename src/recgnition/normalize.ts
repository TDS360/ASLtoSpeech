/**
 * Hand-landmark normalization — a 1:1 port of ml/normalize.py.
 *
 * Training samples and live predictions MUST be preprocessed identically,
 * otherwise the classifier silently degrades. The Python collector and this
 * browser translator therefore share the exact same maths:
 *
 *   1. Translate every point so the wrist (landmark 0) becomes the origin,
 *      making the prediction independent of WHERE the hand is in frame.
 *   2. Divide by the wrist -> middle-finger-MCP distance (landmark 9), making
 *      it independent of HOW FAR the hand is from the camera.
 *
 * Returns a flat array of 63 floats: [x0, y0, z0, x1, y1, z1, ...].
 */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export function normalizeLandmarks(handLandmarks: Landmark[]): number[] {
  const wrist = handLandmarks[0];
  const midPoint = handLandmarks[9];
  if (!wrist || !midPoint) return [];
  const shifted = handLandmarks.map((lm) => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z,
  }));

  const mid = shifted[9];
  if (!mid) return [];
  let scale = Math.sqrt(mid.x * mid.x + mid.y * mid.y + mid.z * mid.z);
  if (scale === 0) scale = 1e-6;

  const out: number[] = [];
  for (const p of shifted) {
    out.push(p.x / scale, p.y / scale, p.z / scale);
  }
  return out;
}
