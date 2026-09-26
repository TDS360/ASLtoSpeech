import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "idle" | "starting" | "ready" | "error";

export interface CameraError {
  title: string;
  detail: string;
  canRetry: boolean;
}

/** Turns browser media errors into something a person can act on. */
function describeError(error: unknown): CameraError {
  const name = (error as DOMException | undefined)?.name ?? "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "Camera permission was blocked",
        detail:
          "Your browser is not letting this page use the camera. Open the padlock icon in the address bar, allow camera access, then try again.",
        canRetry: true,
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return {
        title: "No camera found",
        detail: "No usable camera is connected. Plug one in or pick a different device, then try again.",
        canRetry: true,
      };
    case "NotReadableError":
      return {
        title: "The camera is already in use",
        detail:
          "Another app or browser tab is using the camera. Close it (video calls are the usual culprit) and try again.",
        canRetry: true,
      };
    default:
      return {
        title: "The camera could not be started",
        detail: "Something went wrong reaching the camera. Check that a camera is connected and try again.",
        canRetry: true,
      };
  }
}

export function useCamera(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<CameraError | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | undefined>(undefined);
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, [videoRef]);

  const start = useCallback(
    async (requestedDeviceId?: string) => {
      setStatus("starting");
      setError(null);
      try {
        if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
          throw new DOMException("unsupported", "NotFoundError");
        }
        streamRef.current?.getTracks().forEach((track) => track.stop());

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...(requestedDeviceId ? { deviceId: { exact: requestedDeviceId } } : {}),
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
          audio: false,
        });
        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }

        const list = await navigator.mediaDevices.enumerateDevices();
        setDevices(list.filter((d) => d.kind === "videoinput"));
        setDeviceId(requestedDeviceId ?? stream.getVideoTracks()[0]?.getSettings().deviceId);
        setStatus("ready");
      } catch (err) {
        setError(describeError(err));
        setStatus("error");
      }
    },
    [videoRef],
  );

  useEffect(() => () => stop(), [stop]);

  return { status, error, devices, deviceId, start, stop };
}
