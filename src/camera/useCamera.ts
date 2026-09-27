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
      return {
        title: "Camera access was denied",
        detail:
          "Allow camera access for this site in your browser's site settings, then reload. Also check Windows Settings > Privacy & security > Camera.",
        canRetry: true,
      };
    case "SecurityError":
      return {
        title: "This page cannot access the camera",
        detail:
          "Open the app at localhost or use HTTPS. Camera access is blocked on insecure pages.",
        canRetry: true,
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return {
        title: "No camera found",
        detail: "Connect a camera or choose another camera, then try again.",
        canRetry: true,
      };
    case "NotReadableError":
      return {
        title: "The camera is already in use",
        detail: "Close other apps or browser tabs using the camera, then retry.",
        canRetry: true,
      };
    default:
      return {
        title: "The camera could not be started",
        detail: "Check camera permissions and that a camera is connected.",
        canRetry: true,
      };
  }
}
// ...existing code...
        if (typeof window !== "undefined" && !window.isSecureContext) {
          throw new DOMException("Camera access requires HTTPS or localhost.", "SecurityError");
        }

        if (!navigator.mediaDevices?.getUserMedia) {
          throw new DOMException("Camera access is unavailable in this browser.", "NotSupportedError");
        }
// ...existing code...
export function useCamera(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<CameraError | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | undefined>(undefined);
  const streamRef = useRef<MediaStream | null>(null);
  const requestRef = useRef(0);

  const stop = useCallback(() => {
    requestRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStatus("idle");
  }, [videoRef]);

// ...existing code...
  const start = useCallback(
    async (requestedDeviceId?: string): Promise<boolean> => {
      const requestId = ++requestRef.current;
      setStatus("starting");
      setError(null);

      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new DOMException(
            "Camera access requires HTTPS or localhost.",
            "SecurityError",
          );
        }

        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...(requestedDeviceId
              ? { deviceId: { exact: requestedDeviceId } }
              : {}),
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
          audio: false,
        });

        if (requestId !== requestRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return false;
        }

        streamRef.current = stream;

        const video = videoRef.current;
        if (!video) {
          throw new Error("Video element is unavailable.");
        }

        video.muted = true;
        video.playsInline = true;
        video.srcObject = stream;
        await video.play();

        const list = await navigator.mediaDevices.enumerateDevices();
        if (requestId !== requestRef.current) return false;

        setDevices(list.filter((device) => device.kind === "videoinput"));
        setDeviceId(
          requestedDeviceId ?? stream.getVideoTracks()[0]?.getSettings().deviceId,
        );
        setStatus("ready");
        return true;
      } catch (err) {
        if (requestId !== requestRef.current) return false;

        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        if (videoRef.current) videoRef.current.srcObject = null;

        setError(describeError(err));
        setStatus("error");
        return false;
      }
    },
    [videoRef],
  );
// ...existing code...
  useEffect(() => () => stop(), [stop]);

  return { status, error, devices, deviceId, start, stop };
}