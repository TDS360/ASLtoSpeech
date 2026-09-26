/**
 * MediaPipe HandLandmarker loader.
 *
 * The detector model (models/hand_landmarker.task) is the one already in the
 * repository, served locally. The WASM runtime comes from the pinned CDN
 * build of @mediapipe/tasks-vision.
 */
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL = "/models/hand_landmarker.task";

let instance: HandLandmarker | null = null;
let pending: Promise<HandLandmarker> | null = null;

export async function getHandLandmarker(): Promise<HandLandmarker> {
  if (instance) return instance;
  if (pending) return pending;

  pending = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    const landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    instance = landmarker;
    return landmarker;
  })();

  try {
    return await pending;
  } catch (error) {
    pending = null;
    throw error;
  }
}

export const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** Synchronous access for the per-frame loop (null until loaded). */
export function getCachedHandLandmarker(): HandLandmarker | null {
  return instance;
}
