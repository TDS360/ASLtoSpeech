import { useEffect, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

import { useCamera } from "@/camera/useCamera";
import { useTranslator } from "@/recgnition/useTranslator";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [autoSpeak, setAutoSpeak] = useState(true);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const camera = useCamera(videoRef);
  const translator = useTranslator(videoRef, canvasRef);

  const speak = (text: string) => {
    const cleanText = text.trim();
    if (!cleanText || typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    utterance.pitch = 1;

    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const startSystem = async () => {
    const cameraOk = await camera.start(camera.deviceId);

    if (!cameraOk && camera.status !== "ready") {
      return;
    }

    const translatorOk = await translator.start();

    if (translatorOk) {
      setCameraStarted(true);
    }
  };

  const stopSystem = () => {
    translator.stop();
    camera.stop();
    setCameraStarted(false);

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  };

  const speakSentence = () => {
    if (translator.sentence) {
      speak(translator.sentence);
    }
  };

  // Automatically speak when a completed word changes.
  // This only happens after the 2-second hand-down pause.
  useEffect(() => {
    if (!autoSpeak || translator.words.length === 0) return;

    const latestWord = translator.words[translator.words.length - 1];
    if (!latestWord) return;

    speak(translator.words.join(" "));
  }, [translator.words, autoSpeak]);

  return (
    <main className="asl-app">
      <header className="topbar">
        <div>
          <h1>ASLtoSpeech</h1>
          <p>American Sign Language Fingerspelling Translator</p>
        </div>

        <div className="status">
          <span
            className={`status-dot ${
              camera.status === "ready" && translator.modelReady
                ? "online"
                : "offline"
            }`}
          />
          {camera.status === "ready" && translator.modelReady
            ? "Ready"
            : "Not Ready"}
        </div>
      </header>

      <section className="main-grid">
        <div className="camera-panel">
          <div className="panel-header">
            <h2>Camera</h2>

            <div className="camera-status">
              {camera.status === "ready"
                ? "Camera connected"
                : camera.status === "starting"
                  ? "Starting camera..."
                  : "Camera off"}
            </div>
          </div>

          <div className="camera-view">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="camera-video"
            />

            <canvas
              ref={canvasRef}
              className="camera-overlay"
            />

            {!cameraStarted && (
              // ...existing code...
            <div className="sentence-value">
              {translator.sentence || "Sign letters to build your sentence."}
            </div>
// ...existing code...
            )}
          </div>

          {camera.error && (
            <div className="error-box">
              <strong>{camera.error.title}</strong>
              <span>{camera.error.detail}</span>
            </div>
          )}

          {translator.modelError && (
            <div className="error-box">
              <strong>Recognition model error</strong>
              <span>{translator.modelError}</span>
            </div>
          )}

          <div className="controls">
            {!cameraStarted ? (
              <button onClick={startSystem} className="primary-button">
                Start Translator
              </button>
            ) : (
              <button onClick={stopSystem} className="secondary-button">
                Stop Translator
              </button>
            )}

            <button
              onClick={translator.togglePause}
              disabled={!cameraStarted}
              className="secondary-button"
            >
              {translator.paused ? "Resume" : "Pause"}
            </button>

            <button
              onClick={translator.clearSentence}
              className="secondary-button"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="translation-panel">
          <div className="translation-header">
            <div>
              <h2>Translation</h2>
              <p>Hold a letter steady to add it.</p>
            </div>

            <div className="confidence">
              {translator.snapshot.letter ? (
                <>
                  <span>Confidence</span>
                  <strong>
                    {(translator.snapshot.confidence * 100).toFixed(0)}%
                  </strong>
                </>
              ) : (
                <>
                  <span>Confidence</span>
                  <strong>--</strong>
                </>
              )}
            </div>
          </div>

          <div className="letter-card">
            <span className="label">CURRENT LETTER</span>

            <div className="current-letter">
              {translator.snapshot.letter ?? "—"}
            </div>

            <div className="letter-state">
              {translator.snapshot.liveState === "holding" &&
                "Hold steady..."}

              {translator.snapshot.liveState === "committed" &&
                "Letter added"}

              {translator.snapshot.liveState === "uncertain" &&
                "Low confidence"}

              {translator.snapshot.liveState === "idle" &&
                "Waiting for hand"}
            </div>

            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: `${translator.snapshot.holdProgress * 100}%`,
                }}
              />
            </div>
          </div>

          <div className="word-card">
            <span className="label">LETTERS / CURRENT WORD</span>

            <div className="word-value">
              {translator.letters || "—"}
            </div>
          </div>

          <div className="sentence-card">
            <div className="sentence-card-header">
              <span className="label">SENTENCE</span>

              <button
                onClick={speakSentence}
                disabled={!translator.sentence || speaking}
                className="speak-button"
              >
                {speaking ? "Speaking..." : "Speak Sentence"}
              </button>
            </div>

            <div className="sentence-value">
              {translator.sentence || "Your sentence will appear here."}
            </div>
          </div>

          <div className="settings-card">
            <div className="setting-row">
              <div>
                <strong>Automatic speech</strong>
                <span>Speak after a word is completed.</span>
              </div>

              <button
                onClick={() => setAutoSpeak((value) => !value)}
                className={`toggle ${autoSpeak ? "active" : ""}`}
                aria-pressed={autoSpeak}
              >
                <span />
              </button>
            </div>

            <div className="setting-row">
              <div>
                <strong>Hand-down space</strong>
                <span>2 seconds without a hand creates a space.</span>
              </div>

              <strong className="setting-value">2.0s</strong>
            </div>

            <div className="setting-row">
              <div>
                <strong>Hands detected</strong>
                <span>Using the first detected hand for recognition.</span>
              </div>

              <strong className="setting-value">
                {translator.snapshot.handsDetected}
              </strong>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <span>
          {translator.modelReady
            ? "Hand recognition model loaded"
            : "Loading recognition model when started"}
        </span>

        <span>
          FPS: {translator.stats.fps.toFixed(0)} · Detection:{" "}
          {translator.stats.detectionMs.toFixed(1)} ms
        </span>
      </footer>
    </main>
  );
}