"""
Camera-independent core of the translator.

Everything here is pure Python with no OpenCV / MediaPipe / audio
dependency, so it can be unit-tested (see ml/tests/) without a webcam:

  SignStabilizer  - turns a noisy per-frame prediction stream into
                    deliberate letter events (majority vote, time-based
                    hold, repeat lock, confidence threshold).
  polish_sentence - rule-based grammar/punctuation cleanup. It never
                    invents content words: every signed word survives,
                    only glue ("am", "the", "to") and punctuation are
                    added, and the RAW gloss is always kept alongside.
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------
# Temporal smoothing / debouncing
# ---------------------------------------------------------------------
@dataclass
class StabilizerEvent:
    kind: str                      # "idle" | "uncertain" | "holding" | "commit"
    letter: Optional[str] = None
    confidence: float = 0.0
    progress: float = 0.0          # 0..1 hold progress (for a UI bar)


class SignStabilizer:
    """
    - Low-confidence frames are reported as "uncertain" and never count.
    - A majority vote over the last `window_size` confident frames kills
      single-frame flicker.
    - The voted letter must stay the winner for `hold_seconds` (wall
      clock, so a slow Raspberry Pi behaves like a fast desktop).
    - After a commit the letter is LOCKED: it can't fire again until the
      hand leaves the frame or a different letter is held, which stops
      "HELLO" turning into "HHHHEEELLLL". Double letters (the LL in
      HELLO) work by briefly breaking the handshape or dropping the hand.
    - `uncertain_after` consecutive uncertain frames while holding a hand
      up are surfaced so the UI can say "Uncertain -- please sign again".
    """

    def __init__(self, confidence_threshold=0.6, hold_seconds=0.2,
                 window_size=5, uncertain_after=8):
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.window = deque(maxlen=max(1, int(window_size)))
        self.uncertain_after = uncertain_after
        self._held = None
        self._held_since = None
        self._locked = None
        self._uncertain_run = 0

    def hand_lost(self):
        self.window.clear()
        self._held = None
        self._held_since = None
        self._locked = None
        self._uncertain_run = 0
        return StabilizerEvent("idle")

    reset = hand_lost

    @property
    def persistently_uncertain(self):
        return self._uncertain_run >= self.uncertain_after

    def push(self, letter, confidence, now):
        if letter is None or confidence < self.confidence_threshold:
            self._uncertain_run += 1
            self.window.append(None)
        else:
            self._uncertain_run = 0
            self.window.append(letter)

        votes = [l for l in self.window if l is not None]
        if not votes:
            self._held = None
            self._held_since = None
            return StabilizerEvent("uncertain", letter, confidence)

        winner, count = Counter(votes).most_common(1)[0]
        # Require a real majority of the window, not just a plurality of 1.
        if count < max(1, (self.window.maxlen + 1) // 2) and len(self.window) == self.window.maxlen:
            return StabilizerEvent("uncertain", letter, confidence)

        if winner != self._held:
            self._held = winner
            self._held_since = now
            if self._locked is not None and winner != self._locked:
                self._locked = None  # a different stable letter releases the lock

        if winner == self._locked:
            return StabilizerEvent("holding", winner, confidence, 1.0)

        held_for = now - self._held_since
        progress = 1.0 if self.hold_seconds <= 0 else min(1.0, held_for / self.hold_seconds)
        if held_for >= self.hold_seconds:
            self._locked = winner
            return StabilizerEvent("commit", winner, confidence, 1.0)
        return StabilizerEvent("holding", winner, confidence, progress)


# ---------------------------------------------------------------------
# Sentence cleanup (rule-based, never invents signs)
# ---------------------------------------------------------------------
QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which"}
SUBJECT_PRONOUNS = {"i", "you", "he", "she", "we", "they", "it"}
PROGRESSIVE = {
    "go": "going", "eat": "eating", "drink": "drinking", "work": "working",
    "study": "studying", "learn": "learning", "read": "reading",
    "write": "writing", "walk": "walking", "drive": "driving",
    "run": "running", "sleep": "sleeping", "wait": "waiting",
    "come": "coming", "play": "playing", "cook": "cooking",
    "look": "looking", "talk": "talking",
}
COUNTABLE_NOUNS = {
    "store", "school", "doctor", "bathroom", "hospital", "library", "bus",
    "car", "train", "house", "office", "restaurant", "park", "room", "book",
    "phone", "computer", "teacher", "nurse", "movie", "street", "door",
    "window", "chair", "table", "letter", "key",
}
DESTINATION_VERBS = {"go", "going", "come", "coming", "drive", "driving",
                     "walk", "walking"}
DETERMINERS = {"the", "a", "an", "my", "your", "his", "her", "our", "their", "to"}
GREETINGS = {"hello", "hi", "hey", "thanks", "bye", "goodbye"}
# ASL questions omit "are"/"do": HOW YOU -> "how are you", WHERE YOU GO -> ...
QUESTION_COPULA_PRONOUNS = {"you", "we", "they"}


def _copula(subject):
    if subject == "i":
        return "am"
    if subject in ("he", "she", "it"):
        return "is"
    return "are"


def polish_sentence(words):
    """Returns (raw_gloss, sentence, notes)."""
    source = [w.lower() for w in words if w]
    raw = " ".join(source).upper()
    notes = []
    if not source:
        return raw, "", notes

    greeting = None
    if source[0] in GREETINGS and len(source) > 1:
        greeting = source[0]
        source = source[1:]

    is_question = source[0] in QUESTION_WORDS or (
        source[0] == "you" and ("help" in source or "understand" in source))

    out = []
    for i, word in enumerate(source):
        prev = out[-1] if out else None
        nxt = source[i + 1] if i + 1 < len(source) else None

        # HOW YOU -> how are you ; WHERE YOU GO -> where are you going
        if prev in QUESTION_WORDS and word in SUBJECT_PRONOUNS and len(out) == 1:
            cop = _copula(word)
            out += [cop, word]
            notes.append(f'Added "{cop}" to form a question')
            continue
        if word in SUBJECT_PRONOUNS and nxt in PROGRESSIVE and not (prev in ("am", "is", "are")):
            cop = _copula(word)
            out += [word, cop]
            notes.append(f'Added "{cop}" after "{word}"')
            continue
        if word in PROGRESSIVE and prev in ("am", "is", "are") or (
                word in PROGRESSIVE and len(out) >= 2 and out[-2] in ("am", "is", "are")
                and out[-1] in SUBJECT_PRONOUNS):
            out.append(PROGRESSIVE[word])
            notes.append(f'"{word}" -> "{PROGRESSIVE[word]}"')
            continue
        if word in COUNTABLE_NOUNS and prev in DESTINATION_VERBS:
            out += ["to", "the", word]
            notes.append(f'Added "to the" before "{word}"')
            continue
        if word in COUNTABLE_NOUNS and prev and prev not in DETERMINERS:
            out += ["the", word]
            notes.append(f'Added "the" before "{word}"')
            continue
        out.append(word)

    body = " ".join("I" if w == "i" else w for w in out)
    body = body[0].upper() + body[1:]
    body += "?" if is_question else "."
    if is_question:
        notes.append("Punctuated as a question")

    if greeting:
        sentence = f"{greeting.capitalize()}, {body[0].lower() + body[1:] if not body.startswith('I ') else body}"
    else:
        sentence = body
    return raw, sentence, notes
