import { useCallback, useEffect, useRef, useState } from "react";

import { getCachedHandLandmarker, getHandLandmarker, HAND_CONNECTIONS } from "@/camera/handLandmarker";
import { classifyLandmarks } from "./classifier";
import { SignStabilizer } from "./stabilizer";
import { correctWord } from "@/sentence/dictionary";
import { polishSentence } from "@/sentence/polish";

export interface TranslatorSettings {
  confidenceThreshold: number;
  holdFrames: number;
  windowSize: number;
  /** Silence (no hand) that ends the current fingerspelled word, in ms. */
  wordPauseMs: number;
  /** Target processing rate. The video still renders at full speed. */
  targetFps: number;
  drawSkeleton: boolean;
}

export const DEFAULT_SETTINGS: TranslatorSettings = {
  confidenceThreshold: 0.7,
  holdFrames: 6,
  windowSize: 7,
  wordPauseMs: 1100,
  targetFps: 15,
  drawSkeleton: true,
};

export type LiveState = "idle" | "holding" | "uncertain" | "committed";

export interface TranslatorStats {
  fps: number;
  inferenceMs: number;
  detectionMs: number;
}

export interface TranslatorSnapshot {
  handsDetected: number;
  brightness: number;
  letter: string | null;
  confidence: number;
  alternative: { letter: string; confidence: number } | null;
  liveState: LiveState;
  holdProgress: number;
}

const EMPTY_SNAPSHOT: TranslatorSnapshot = {
  handsDetected: 0,
  brightness: 0,
  letter: null,
  confidence: 0,
  alternative: null,
  liveState: "idle",
  holdProgress: 0,
};

export function useTranslator(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
) {
  const [settings, setSettings] = useState<TranslatorSettings>(DEFAULT_SETTINGS);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<TranslatorSnapshot>(EMPTY_SNAPSHOT);
  const [stats, setStats] = useState<TranslatorStats>({ fps: 0, inferenceMs: 0, detectionMs: 0 });
  const [letters, setLetters] = useState<string>("");
  const [words, setWords] = useState<string[]>([]);

  const settingsRef = useRef(settings);
  const runningRef = useRef(false);
  const pausedRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const lastProcessRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastHandSeenRef = useRef(0);
  const letterBufferRef = useRef("");
  const frameTimesRef = useRef<number[]>([]);
  const stabilizerRef = useRef(
    new SignStabilizer({
      confidenceThreshold: DEFAULT_SETTINGS.confidenceThreshold,
      holdFrames: DEFAULT_SETTINGS.holdFrames,
      windowSize: DEFAULT_SETTINGS.windowSize,
    }),
  );

  useEffect(() => {
    settingsRef.current = settings;
    stabilizerRef.current.setOptions({
      confidenceThreshold: settings.confidenceThreshold,
      holdFrames: settings.holdFrames,
      windowSize: settings.windowSize,
    });
  }, [settings]);

  /** Warm the detector up front so pressing Start feels instant. */
  const loadModel = useCallback(async () => {
    try {
      await getHandLandmarker();
      setModelReady(true);
      setModelError(null);
      return true;
    } catch {
      setModelReady(false);
      setModelError(
        "The hand-detection model could not be loaded. Check your connection and reload the page — recognition is unavailable until it loads.",
      );
      return false;
    }
  }, []);

  const flushWord = useCallback(() => {
    const buffer = letterBufferRef.current;
    if (!buffer) return;
    letterBufferRef.current = "";
    setLetters("");
    const { word } = correctWord(buffer);
    setWords((prev) => [...prev, word]);
  }, []);

  const drawOverlay = useCallback(
    (hands: Array<Array<{ x: number; y: number }>>, width: number, height: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);
      if (!settingsRef.current.drawSkeleton) return;

      const accent = getComputedStyle(canvas).getPropertyValue("--overlay-accent").trim() || "#2dd4bf";
      ctx.lineWidth = Math.max(2, width / 320);
      ctx.strokeStyle = accent;
      ctx.fillStyle = accent;

      for (const hand of hands) {
        for (const [a, b] of HAND_CONNECTIONS) {
          const from = hand[a];
          const to = hand[b];
          if (!from || !to) continue;
          ctx.beginPath();
          ctx.moveTo(from.x * width, from.y * height);
          ctx.lineTo(to.x * width, to.y * height);
          ctx.stroke();
        }
        for (const point of hand) {
          ctx.beginPath();
          ctx.arc(point.x * width, point.y * height, ctx.lineWidth * 1.3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    },
    [canvasRef],
  );

  const sampleBrightness = useCallback((video: HTMLVideoElement) => {
    const sampler = document.createElement("canvas");
    sampler.width = 32;
    sampler.height = 24;
    const ctx = sampler.getContext("2d", { willReadFrequently: true });
    if (!ctx) return 0;
    ctx.drawImage(video, 0, 0, 32, 24);
    const { data } = ctx.getImageData(0, 0, 32, 24);
    let total = 0;
    for (let i = 0; i < data.length; i += 4) {
      total += (data[i] + data[i + 1] + data[i + 2]) / 3;
    }
    return total / (data.length / 4) / 255;
  }, []);

  const loop = useCallback(() => {
    if (!runningRef.current) return;
    rafRef.current = requestAnimationFrame(loop);

    const video = videoRef.current;
    if (!video || video.readyState < 2 || pausedRef.current) return;

    const now = performance.now();
    const minGap = 1000 / settingsRef.current.targetFps;
    if (now - lastProcessRef.current < minGap) return; // frame skipping keeps CPU low
    if (video.currentTime === lastVideoTimeRef.current) return;
    lastProcessRef.current = now;
    lastVideoTimeRef.current = video.currentTime;

    const landmarker = getCachedHandLandmarker();
    if (!landmarker) return;

    let result;
    const detectStart = performance.now();
    try {
      result = landmarker.detectForVideo(video, now);
    } catch {
      return; // a single bad frame must never take the app down
    }
    const detectionMs = performance.now() - detectStart;

    const hands = result.landmarks ?? [];
    drawOverlay(hands, video.videoWidth || 640, video.videoHeight || 480);

    frameTimesRef.current.push(now);
    if (frameTimesRef.current.length > 30) frameTimesRef.current.shift();
    const span = frameTimesRef.current[frameTimesRef.current.length - 1] - frameTimesRef.current[0];
    const fps = span > 0 ? ((frameTimesRef.current.length - 1) / span) * 1000 : 0;

    const brightness = sampleBrightness(video);

    if (hands.length === 0) {
      stabilizerRef.current.handLost();
      if (
        letterBufferRef.current &&
        now - lastHandSeenRef.current > settingsRef.current.wordPauseMs
      ) {
        flushWord();
      }
      setSnapshot((prev) => ({
        ...prev,
        handsDetected: 0,
        brightness,
        liveState: "idle",
        holdProgress: 0,
      }));
      setStats({ fps, inferenceMs: 0, detectionMs });
      return;
    }

    lastHandSeenRef.current = now;

    // Only the first hand feeds the classifier: the training data is
    // single-hand, so guessing from two hands would be dishonest.
    const prediction = classifyLandmarks(hands[0]);
    if (!prediction) return;

    const event = stabilizerRef.current.push(prediction.letter, prediction.confidence);
    let liveState: LiveState = "holding";
    let holdProgress = 0;

    if (event.type === "uncertain") liveState = "uncertain";
    else if (event.type === "commit") {
      liveState = "committed";
      letterBufferRef.current += event.letter;
      setLetters(letterBufferRef.current);
    } else if (event.type === "holding") holdProgress = event.progress;

    setSnapshot({
      handsDetected: hands.length,
      brightness,
      letter: prediction.letter,
      confidence: prediction.confidence,
      alternative: prediction.alternative,
      liveState,
      holdProgress,
    });
    setStats({ fps, inferenceMs: prediction.inferenceMs, detectionMs });
  }, [drawOverlay, flushWord, sampleBrightness, videoRef]);

  const start = useCallback(async () => {
    const ok = modelReady || (await loadModel());
    if (!ok) return false;
    runningRef.current = true;
    pausedRef.current = false;
    lastHandSeenRef.current = performance.now();
    setRunning(true);
    setPaused(false);
    if (rafRef.current === null) rafRef.current = requestAnimationFrame(loop);
    return true;
  }, [loadModel, loop, modelReady]);

  const stop = useCallback(() => {
    runningRef.current = false;
    pausedRef.current = false;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    stabilizerRef.current.reset();
    setRunning(false);
    setPaused(false);
    setSnapshot(EMPTY_SNAPSHOT);
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }, [canvasRef]);

  const togglePause = useCallback(() => {
    pausedRef.current = !pausedRef.current;
    stabilizerRef.current.reset();
    setPaused(pausedRef.current);
  }, []);

  const clearSentence = useCallback(() => {
    letterBufferRef.current = "";
    stabilizerRef.current.reset();
    setLetters("");
    setWords([]);
  }, []);

  const backspace = useCallback(() => {
    if (letterBufferRef.current) {
      letterBufferRef.current = letterBufferRef.current.slice(0, -1);
      setLetters(letterBufferRef.current);
      return;
    }
    setWords((prev) => prev.slice(0, -1));
  }, []);

  useEffect(() => () => {
    runningRef.current = false;
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
  }, []);

  const allWords = letters ? [...words, letters.toLowerCase()] : words;
  const { raw, sentence, notes } = polishSentence(allWords);

  return {
    settings,
    setSettings,
    running,
    paused,
    modelReady,
    modelError,
    snapshot,
    stats,
    letters,
    words,
    raw,
    sentence,
    notes,
    loadModel,
    start,
    stop,
    togglePause,
    clearSentence,
    backspace,
    endWord: flushWord,
  };
}
