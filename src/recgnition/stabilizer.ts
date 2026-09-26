/**
 * Temporal smoothing + debouncing.
 *
 * A raw per-frame classifier fires ~15 times a second, so signing "HELLO"
 * once would otherwise append dozens of letters. This stabilizer turns a
 * noisy prediction stream into discrete, deliberate sign events:
 *
 *   - a majority vote over a short rolling window kills single-frame noise,
 *   - the winning letter must be held for `holdFrames` consecutive frames,
 *   - after a letter is emitted it is LOCKED: the same letter cannot fire
 *     again until the hand leaves the frame or a different letter is held
 *     (this is what stops HELLO HELLO HELLO HELLO),
 *   - predictions below the confidence threshold are reported as uncertain
 *     and never committed.
 */

export interface StabilizerOptions {
  /** Minimum probability before a prediction may be committed (0–1). */
  confidenceThreshold: number;
  /** Consecutive agreeing frames required to commit a letter. */
  holdFrames: number;
  /** Rolling window size used for the majority vote. */
  windowSize: number;
}

export type StabilizerEvent =
  | { type: "idle" }
  | { type: "uncertain"; letter: string; confidence: number }
  | { type: "holding"; letter: string; confidence: number; progress: number }
  | { type: "commit"; letter: string; confidence: number };

export class SignStabilizer {
  private window: string[] = [];
  private heldLetter: string | null = null;
  private heldFrames = 0;
  private lockedLetter: string | null = null;

  constructor(private options: StabilizerOptions) {}

  setOptions(options: Partial<StabilizerOptions>) {
    this.options = { ...this.options, ...options };
  }

  /** Call when no hand is visible — releases the repeat lock. */
  handLost(): StabilizerEvent {
    this.window = [];
    this.heldLetter = null;
    this.heldFrames = 0;
    this.lockedLetter = null;
    return { type: "idle" };
  }

  reset() {
    this.handLost();
  }

  push(letter: string, confidence: number): StabilizerEvent {
    if (confidence < this.options.confidenceThreshold) {
      this.window = [];
      this.heldLetter = null;
      this.heldFrames = 0;
      return { type: "uncertain", letter, confidence };
    }

    this.window.push(letter);
    if (this.window.length > this.options.windowSize) this.window.shift();

    // Majority vote across the rolling window.
    const counts = new Map<string, number>();
    for (const l of this.window) counts.set(l, (counts.get(l) ?? 0) + 1);
    let winner = letter;
    let winnerCount = 0;
    for (const [l, c] of counts) {
      if (c > winnerCount) {
        winner = l;
        winnerCount = c;
      }
    }

    if (winner !== this.heldLetter) {
      this.heldLetter = winner;
      this.heldFrames = 1;
    } else {
      this.heldFrames += 1;
    }

    // A different stable letter clears the repeat lock, so "LL" still works
    // when the signer breaks the handshape in between.
    if (this.lockedLetter && winner !== this.lockedLetter && this.heldFrames >= 2) {
      this.lockedLetter = null;
    }

    if (winner === this.lockedLetter) {
      return { type: "holding", letter: winner, confidence, progress: 1 };
    }

    if (this.heldFrames >= this.options.holdFrames) {
      this.lockedLetter = winner;
      this.heldFrames = 0;
      return { type: "commit", letter: winner, confidence };
    }

    return {
      type: "holding",
      letter: winner,
      confidence,
      progress: Math.min(1, this.heldFrames / this.options.holdFrames),
    };
  }
}
