import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Optional reply channel: microphone -> text, so a hearing person can answer
 * on screen. Entirely self-contained — if the browser has no speech
 * recognition, the ASL side of the app is unaffected.
 */
interface RecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((event: never) => void) | null;
  onerror: ((event: never) => void) | null;
  onend: (() => void) | null;
}

export function useSpeechToText() {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<RecognitionLike | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const win = window as unknown as Record<string, unknown>;
    const Ctor = (win["SpeechRecognition"] ?? win["webkitSpeechRecognition"]) as
      | (new () => RecognitionLike)
      | undefined;
    if (Ctor) setSupported(true);
    return () => {
      try {
        recognitionRef.current?.stop();
      } catch {
        /* already stopped */
      }
    };
  }, []);

  const stop = useCallback(() => {
    try {
      recognitionRef.current?.stop();
    } catch {
      /* already stopped */
    }
    setListening(false);
  }, []);

  const start = useCallback(() => {
    if (typeof window === "undefined") return;
    const win = window as unknown as Record<string, unknown>;
    const Ctor = (win["SpeechRecognition"] ?? win["webkitSpeechRecognition"]) as
      | (new () => RecognitionLike)
      | undefined;
    if (!Ctor) {
      setError("This browser cannot listen to speech. Chrome or Edge on desktop works best.");
      return;
    }

    setError(null);
    const recognition = new Ctor();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onresult = ((event: {
      resultIndex: number;
      results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
    }) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const alternative = result?.[0];
        if (!alternative) continue;
        const text = alternative.transcript;
        if (result.isFinal) finalText += text;
        else interimText += text;
      }
      if (finalText) setTranscript((prev) => (prev ? `${prev} ${finalText.trim()}` : finalText.trim()));
      setInterim(interimText);
    }) as unknown as (event: never) => void;

    recognition.onerror = ((event: { error?: string }) => {
      const code = event?.error;
      setError(
        code === "not-allowed"
          ? "Microphone permission was blocked. Allow the microphone in your browser and try again."
          : "Listening stopped unexpectedly. Try again.",
      );
      setListening(false);
    }) as unknown as (event: never) => void;

    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
    } catch {
      setError("Listening could not start. Try again in a moment.");
    }
  }, []);

  const clear = useCallback(() => {
    setTranscript("");
    setInterim("");
  }, []);

  return { supported, listening, transcript, interim, error, start, stop, clear };
}
