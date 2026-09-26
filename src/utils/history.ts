import { useCallback, useEffect, useState } from "react";

/**
 * Translation history.
 *
 * Text only, stored locally in this browser. No video, no audio, no images
 * are ever recorded or uploaded — see the privacy note in the UI.
 */
export interface HistoryEntry {
  id: string;
  at: number;
  raw: string;
  sentence: string;
}

const STORAGE_KEY = "asl-speech-history-v1";
const MAX_ENTRIES = 50;

function read(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]");
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => setEntries(read()), []);

  const persist = useCallback((next: HistoryEntry[]) => {
    setEntries(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* storage full or blocked — history simply won't persist */
    }
  }, []);

  const add = useCallback(
    (raw: string, sentence: string) => {
      const entry: HistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        at: Date.now(),
        raw,
        sentence,
      };
      persist([entry, ...read()].slice(0, MAX_ENTRIES));
    },
    [persist],
  );

  const remove = useCallback((id: string) => persist(read().filter((e) => e.id !== id)), [persist]);
  const clear = useCallback(() => persist([]), [persist]);

  return { entries, add, remove, clear };
}
