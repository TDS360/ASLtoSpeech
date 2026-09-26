/**
 * Small offline dictionary used for gentle fingerspelling correction.
 *
 * This is the browser equivalent of the `pyspellchecker` step in the original
 * Python translator. It only ever repairs a spelled word by AT MOST one edit,
 * and only when exactly one dictionary word is that close — it never invents
 * a word out of nothing.
 */

export const COMMON_WORDS: string[] = [
  "a","about","after","again","all","also","always","am","an","and","any","are","ask","at","away",
  "back","bad","bathroom","be","because","bed","been","before","best","better","big","book","both",
  "boy","bread","bring","brother","bus","but","buy","by","call","can","car","cat","chair","child",
  "city","class","clean","close","coffee","cold","come","computer","cook","could","country","cup",
  "dad","dance","day","deaf","did","dinner","do","doctor","dog","door","down","drink","drive","eat",
  "email","end","enough","every","eye","face","family","far","fast","father","feel","find","fine",
  "finish","fire","first","fish","food","for","friend","from","fun","get","girl","give","glad","go",
  "good","goodbye","got","great","green","hand","happy","hard","has","hat","hate","have","he","head",
  "health","hear","hello","help","her","here","hi","him","his","home","hope","hospital","hot","hour",
  "house","how","hungry","hurt","i","ice","idea","if","in","is","it","job","juice","jump","just",
  "keep","key","kid","kind","know","lake","large","last","late","learn","leave","left","let","letter",
  "library","life","light","like","listen","little","live","long","look","lost","lot","love","lunch",
  "mad","make","man","many","may","maybe","me","meat","medicine","meet","milk","mine","minute","miss",
  "mom","money","month","more","morning","most","mother","move","movie","much","music","must","my",
  "name","near","need","never","new","next","nice","night","no","nod","not","now","nurse","of","off",
  "office","often","ok","old","on","one","only","open","or","order","other","our","out","over","pain",
  "paper","park","pay","pen","people","phone","place","play","please","police","poor","practice",
  "pretty","price","problem","put","question","quick","quiet","rain","read","ready","really","red",
  "remember","repeat","rest","restaurant","ride","right","room","run","sad","safe","same","say",
  "school","sea","see","sell","send","she","shop","short","should","show","sick","sign","sister","sit",
  "sleep","slow","small","smile","snow","so","some","sorry","speak","start","stay","still","stop",
  "store","story","street","strong","student","study","sun","sure","table","take","talk","tall","tea",
  "teach","teacher","tell","thank","thanks","that","the","their","them","then","there","these","they",
  "thing","think","this","those","time","tired","to","today","together","tomorrow","too","town","train",
  "travel","tree","try","turn","two","under","understand","up","us","use","very","wait","walk","want",
  "warm","was","watch","water","way","we","wear","week","welcome","well","were","what","when","where",
  "which","while","white","who","why","will","win","window","wish","with","woman","word","work","world",
  "would","write","wrong","year","yes","yesterday","you","young","your",
];

const WORD_SET = new Set(COMMON_WORDS);

function editDistanceAtMostOne(a: string, b: string): boolean {
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0;
  let j = 0;
  let edits = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      i++;
      j++;
      continue;
    }
    if (++edits > 1) return false;
    if (a.length > b.length) i++;
    else if (a.length < b.length) j++;
    else {
      i++;
      j++;
    }
  }
  return edits + (a.length - i) + (b.length - j) <= 1;
}

export interface CorrectionResult {
  word: string;
  corrected: boolean;
  original: string;
}

/**
 * Returns the word unchanged unless exactly one dictionary word is a single
 * edit away. Ambiguous cases are left as-signed, on purpose.
 */
export function correctWord(raw: string): CorrectionResult {
  const lower = raw.toLowerCase();
  if (lower.length < 3 || WORD_SET.has(lower)) {
    return { word: lower, corrected: false, original: lower };
  }
  const matches = COMMON_WORDS.filter((w) => editDistanceAtMostOne(lower, w));
  const match = matches[0];
  if (matches.length === 1 && match !== undefined) {
    return { word: match, corrected: true, original: lower };
  }
  return { word: lower, corrected: false, original: lower };
}
