import { InvalidInputError } from "../errors.js";
import { normalizeText } from "./normalize.js";

/** Characters that end a sentence in the languages VoxShot targets. */
const TERMINATORS = /[.!?。！？…]/;

/** Characters that are acceptable places to break an over-long sentence. */
const BREAKABLE = /[,、，;；:：\s]/;

/** Default synthesis chunk size, tuned to keep per-chunk latency low. */
const DEFAULT_MAX_LENGTH = 120;

export interface SplitSentencesOptions {
  /**
   * Hard upper bound on the length of a returned chunk. Sentences longer than
   * this are broken apart, preferring comma / whitespace boundaries.
   *
   * @defaultValue 120
   */
  maxLength?: number;
  /**
   * Chunks shorter than this are merged with their neighbours, as long as the
   * merge stays within `maxLength`.
   *
   * Very short prompts ("はい", "なぜ?") make neural TTS models unstable, so
   * engines that suffer from it can ask for a floor. `0` disables merging.
   *
   * @defaultValue 0
   */
  minLength?: number;
}

/**
 * Split text into synthesis-sized chunks.
 *
 * Each returned chunk is normalized (see {@link normalizeText}), non empty,
 * and never longer than `maxLength`.
 */
export function splitSentences(text: string, options: SplitSentencesOptions = {}): string[] {
  const maxLength = options.maxLength ?? DEFAULT_MAX_LENGTH;
  if (!Number.isFinite(maxLength) || maxLength <= 0) {
    throw new InvalidInputError("maxLength must be a positive finite number.");
  }
  const minLength = options.minLength ?? 0;
  if (!Number.isFinite(minLength) || minLength < 0) {
    throw new InvalidInputError("minLength must be a non negative finite number.");
  }

  const chunks: string[] = [];
  for (const line of text.split(/[\r\n]+/)) {
    for (const rawSentence of splitLine(line)) {
      const sentence = normalizeText(rawSentence);
      if (sentence.length > 0) {
        chunks.push(...enforceMaxLength(sentence, maxLength));
      }
    }
  }
  return minLength > 0 ? mergeShortChunks(chunks, minLength, maxLength) : chunks;
}

/**
 * Merge chunks that fall below `minLength` into their neighbours, never
 * producing anything longer than `maxLength`.
 *
 * Only *adjacent short* chunks are merged, so a chunk that already clears the
 * floor is left alone. A trailing short chunk has nothing left to absorb, so
 * it is folded back into its predecessor instead.
 */
function mergeShortChunks(chunks: string[], minLength: number, maxLength: number): string[] {
  const merged: string[] = [];

  for (const chunk of chunks) {
    const previous = merged[merged.length - 1];
    const bothShort =
      previous !== undefined && previous.length < minLength && chunk.length < minLength;

    if (bothShort && (previous as string).length + chunk.length <= maxLength) {
      merged[merged.length - 1] = previous + chunk;
    } else {
      merged.push(chunk);
    }
  }

  const last = merged[merged.length - 1];
  const beforeLast = merged[merged.length - 2];
  if (
    last !== undefined &&
    beforeLast !== undefined &&
    last.length < minLength &&
    beforeLast.length + last.length <= maxLength
  ) {
    merged.splice(merged.length - 2, 2, beforeLast + last);
  }
  return merged;
}

/** Opening brackets / quotes mapped to their closing counterparts. */
const BRACKET_PAIRS: Record<string, string> = {
  "「": "」",
  "『": "』",
  "(": ")",
  "（": "）",
  "[": "]",
  "［": "］",
  "{": "}",
  "｛": "｝",
  "〈": "〉",
  "《": "》",
  "【": "】",
  "“": "”",
  "‘": "’",
};

/**
 * Undirected quotes toggle like a switch. The ASCII single quote is left out
 * on purpose: it doubles as an apostrophe (`don't`), which would otherwise
 * suppress sentence splitting for the rest of the line.
 */
const TOGGLING_QUOTES = new Set(['"']);

/**
 * Split a single line after every run of sentence terminators.
 *
 * Terminators inside brackets or quotes do not end the sentence, so quoted
 * speech like 「はい。そうです。」 stays attached to its surrounding sentence.
 * An unmatched opener simply keeps the rest of the line together; depth can
 * never go negative, so a stray closer is ignored.
 */
function splitLine(line: string): string[] {
  const sentences: string[] = [];
  const closers: string[] = [];
  let start = 0;
  let index = 0;

  while (index < line.length) {
    const char = line[index] as string;

    if (closers[closers.length - 1] === char) {
      closers.pop();
      index += 1;
      continue;
    }
    const closer = BRACKET_PAIRS[char];
    if (closer !== undefined) {
      closers.push(closer);
      index += 1;
      continue;
    }
    if (TOGGLING_QUOTES.has(char)) {
      closers.push(char);
      index += 1;
      continue;
    }

    if (closers.length > 0 || !isTerminator(line, index)) {
      index += 1;
      continue;
    }

    let end = index + 1;
    while (end < line.length && isTerminator(line, end)) {
      end += 1;
    }
    sentences.push(line.slice(start, end));
    start = end;
    index = end;
  }

  if (start < line.length) {
    sentences.push(line.slice(start));
  }
  return sentences;
}

/**
 * A terminator ends a sentence unless it is a decimal point sitting between
 * two digits, which keeps numbers such as `3.14` in one piece.
 */
function isTerminator(line: string, index: number): boolean {
  const char = line[index] as string;
  if (!TERMINATORS.test(char)) {
    return false;
  }
  if (char !== ".") {
    return true;
  }
  return !(/\d/.test(line[index - 1] ?? "") && /\d/.test(line[index + 1] ?? ""));
}

/** Break a sentence into chunks no longer than `maxLength`. */
function enforceMaxLength(sentence: string, maxLength: number): string[] {
  if (sentence.length <= maxLength) {
    return [sentence];
  }

  const atoms = splitAtoms(sentence).flatMap((atom) => hardSplit(atom, maxLength));
  const chunks: string[] = [];
  let current = "";

  for (const atom of atoms) {
    const candidate = current + atom;
    if (current.length > 0 && candidate.trimEnd().length > maxLength) {
      chunks.push(current.trim());
      current = atom;
    } else {
      current = candidate;
    }
  }

  const last = current.trim();
  if (last.length > 0) {
    chunks.push(last);
  }
  return chunks;
}

/**
 * Split a sentence into atoms that each end right after a run of breakable
 * characters, so chunks can be reassembled at natural pauses.
 */
function splitAtoms(sentence: string): string[] {
  const atoms: string[] = [];
  let start = 0;
  let index = 0;

  while (index < sentence.length) {
    if (!BREAKABLE.test(sentence[index] as string)) {
      index += 1;
      continue;
    }

    let end = index + 1;
    while (end < sentence.length && BREAKABLE.test(sentence[end] as string)) {
      end += 1;
    }
    atoms.push(sentence.slice(start, end));
    start = end;
    index = end;
  }

  if (start < sentence.length) {
    atoms.push(sentence.slice(start));
  }
  return atoms;
}

/** Last resort splitter for text that offers no break opportunity. */
function hardSplit(atom: string, maxLength: number): string[] {
  if (atom.length <= maxLength) {
    return [atom];
  }
  const pieces: string[] = [];
  for (let offset = 0; offset < atom.length; offset += maxLength) {
    pieces.push(atom.slice(offset, offset + maxLength));
  }
  return pieces;
}
