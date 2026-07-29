import { InvalidInputError } from "../errors.js";
import { normalizeText } from "./normalize.js";

/** Characters that end a sentence in the languages ZeroVox targets. */
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

  const chunks: string[] = [];
  for (const line of text.split(/[\r\n]+/)) {
    for (const rawSentence of splitLine(line)) {
      const sentence = normalizeText(rawSentence);
      if (sentence.length > 0) {
        chunks.push(...enforceMaxLength(sentence, maxLength));
      }
    }
  }
  return chunks;
}

/** Split a single line after every run of sentence terminators. */
function splitLine(line: string): string[] {
  const sentences: string[] = [];
  let start = 0;
  let index = 0;

  while (index < line.length) {
    if (!isTerminator(line, index)) {
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
