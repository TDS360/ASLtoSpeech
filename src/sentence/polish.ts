/**
 * Sentence-processing layer.
 *
 * STRICT RULE: this layer may only reorder nothing, invent nothing, and add
 * no content words. It is limited to grammatical glue (articles, copulas,
 * verb inflection), capitalisation and punctuation, applied by fixed rules to
 * words the recognition model actually produced. Every raw word survives into
 * the output in some recognisable form, and the UI always shows the raw
 * recognition next to the processed sentence so nothing is hidden.
 */

const QUESTION_WORDS = new Set(["who", "what", "where", "when", "why", "how", "which"]);
const SUBJECT_PRONOUNS = new Set(["i", "you", "he", "she", "we", "they"]);

/** Verbs we know how to inflect. Keys are the base (gloss) form. */
const PROGRESSIVE: Record<string, string> = {
  go: "going",
  eat: "eating",
  drink: "drinking",
  work: "working",
  study: "studying",
  learn: "learning",
  read: "reading",
  write: "writing",
  walk: "walking",
  drive: "driving",
  run: "running",
  sleep: "sleeping",
  wait: "waiting",
  come: "coming",
  play: "playing",
  cook: "cooking",
  look: "looking",
  talk: "talking",
};

/** Nouns that read naturally with a definite/indefinite article. */
const COUNTABLE_NOUNS = new Set([
  "store","school","doctor","bathroom","hospital","library","bus","car","train","house","office",
  "restaurant","park","room","book","phone","computer","teacher","nurse","movie","street","door",
  "window","chair","table","letter","key",
]);

const DESTINATION_VERBS = new Set(["go", "going", "come", "coming", "drive", "driving", "walk", "walking"]);

function copulaFor(subject: string): string {
  if (subject === "i") return "am";
  if (subject === "he" || subject === "she" || subject === "it") return "is";
  return "are";
}

export interface PolishResult {
  raw: string;
  sentence: string;
  /** Plain-language notes on what the rules changed. */
  notes: string[];
}

export function polishSentence(words: string[]): PolishResult {
  const raw = words.join(" ").toUpperCase();
  const notes: string[] = [];

  const source = words.map((w) => w.toLowerCase()).filter(Boolean);
  if (source.length === 0) return { raw: "", sentence: "", notes };

  const out: string[] = [];
  const firstWord = source[0] ?? "";
  let isQuestion = QUESTION_WORDS.has(firstWord);

  for (let i = 0; i < source.length; i++) {
    const word = source[i];
    if (!word) continue;
    const previous = out[out.length - 1];
    const next = source[i + 1];

    // Pronoun + bare verb -> progressive with the right copula ("I GO" -> "I am going").
    if (SUBJECT_PRONOUNS.has(word) && next && PROGRESSIVE[next]) {
      out.push(word, copulaFor(word));
      notes.push(`Added "${copulaFor(word)}" after "${word}"`);
      continue;
    }

    const progressive = PROGRESSIVE[word];
    if (progressive && previous && ["am", "is", "are"].includes(previous)) {
      out.push(progressive);
      notes.push(`"${word}" -> "${progressive}"`);
      continue;
    }

    // Movement verb + place -> insert "to the".
    if (COUNTABLE_NOUNS.has(word) && previous && DESTINATION_VERBS.has(previous)) {
      out.push("to", "the", word);
      notes.push(`Added "to the" before "${word}"`);
      continue;
    }

    // Any other bare countable noun gets an article, unless it already has one.
    if (
      COUNTABLE_NOUNS.has(word) &&
      previous &&
      !["the", "a", "an", "my", "your", "his", "her", "our", "their", "to"].includes(previous)
    ) {
      out.push("the", word);
      notes.push(`Added "the" before "${word}"`);
      continue;
    }

    out.push(word);
  }

  // Yes/no questions such as "YOU HELP ME" read as questions too when they
  // open with a pronoun and contain a request verb.
  if (!isQuestion && firstWord === "you" && (source.includes("help") || source.includes("understand"))) {
    isQuestion = true;
  }

  const text = out
    .map((w, i) => (w === "i" ? "I" : i === 0 ? `${w[0] ?? ""}${w.slice(1)}`.toUpperCase() : w))
    .join(" ");

  const sentence = `${text}${isQuestion ? "?" : "."}`;
  if (isQuestion) notes.push("Punctuated as a question");

  return { raw, sentence, notes };
}
