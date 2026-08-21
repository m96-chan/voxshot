/**
 * Qwen2 byte-level BPE tokenizer for MioTTS-0.6B, ported from `tokenizer.json`.
 *
 * The pipeline mirrors the HF `tokenizers` configuration exactly:
 * NFC normalize → split out added tokens → Qwen2 pre-tokenizer regex →
 * GPT-2 byte-to-unicode map → BPE by merge rank → vocab lookup.
 *
 * No filesystem access here: `loadTokenizer` takes the *parsed* tokenizer.json
 * object, so Node tests read the snapshot file and a browser fetches the same
 * bytes. There is no async work today, but loading stays `async` so a future
 * lazy/streaming source does not ripple through every caller.
 */

// ---------------------------------------------------------------------------
// Special ids (fixed by the model card; asserted against the goldens in tests)

export const ENDOFTEXT_ID = 151643; // <|endoftext|> — pad + secondary eos
export const IM_START_ID = 151644; // <|im_start|>
export const IM_END_ID = 151645; // <|im_end|> — eos
export const SPEECH_TOKEN_BASE = 151669; // <|s_0|>
export const SPEECH_TOKEN_COUNT = 12800; // <|s_0|> .. <|s_12799|>

/** Token id of `<|s_n|>`. Throws on an index outside the codebook. */
export function speechTokenId(n: number): number {
  if (!Number.isInteger(n) || n < 0 || n >= SPEECH_TOKEN_COUNT) {
    throw new Error(`speech token index ${n} outside 0..${SPEECH_TOKEN_COUNT - 1}`);
  }
  return SPEECH_TOKEN_BASE + n;
}

/** Codebook index of a `<|s_n|>` id, or null for any other id. */
export function speechIndexOf(id: number): number | null {
  if (
    !Number.isInteger(id) ||
    id < SPEECH_TOKEN_BASE ||
    id >= SPEECH_TOKEN_BASE + SPEECH_TOKEN_COUNT
  ) {
    return null;
  }
  return id - SPEECH_TOKEN_BASE;
}

// ---------------------------------------------------------------------------
// tokenizer.json shape (only the fields this port consumes)

export interface AddedTokenJson {
  id: number;
  content: string;
}

export interface TokenizerJson {
  added_tokens: AddedTokenJson[];
  model: {
    vocab: Record<string, number>;
    /** tokenizer.json 5.x writes `["a", "b"]`; older files write `"a b"`. */
    merges: (string | [string, string])[];
  };
}

export interface EncodeChatOptions {
  /** Append `<|im_start|>assistant\n` after the user turn. Default true. */
  addGenerationPrompt?: boolean;
  /** Optional system message, prepended as its own ChatML turn. */
  system?: string;
}

export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
  encodeChat(userText: string, opts?: EncodeChatOptions): number[];
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level map
//
// Printable-ish bytes map to themselves; the rest are displaced to 256+n so
// every byte has a *printable* stand-in character and BPE never sees a raw
// space or control byte inside a token piece.

function buildByteMaps(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bytes: number[] = [];
  for (let b = 0x21; b <= 0x7e; b += 1) bytes.push(b);
  for (let b = 0xa1; b <= 0xac; b += 1) bytes.push(b);
  for (let b = 0xae; b <= 0xff; b += 1) bytes.push(b);
  const present = new Set(bytes);
  const byteToChar = new Array<string>(256);
  const charToByte = new Map<string, number>();
  let displaced = 0;
  for (let b = 0; b < 256; b += 1) {
    const code = present.has(b) ? b : 256 + displaced++;
    const ch = String.fromCharCode(code);
    byteToChar[b] = ch;
    charToByte.set(ch, b);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteMaps();

// ---------------------------------------------------------------------------
// Pre-tokenizer
//
// The Qwen2 regex. The reference's Rust-only `(?i:...)` scoped
// case-insensitivity is reproduced by compiling the WHOLE pattern with the
// `i` flag: with `u`, JS applies the same Unicode simple case folding
// fancy-regex does, so U+017F (ſ) folds to 's' — an explicit `[sS]` class
// would miss it. Making the whole pattern case-insensitive changes nothing
// else: outside the contraction group the pattern contains no cased literals,
// only `\p{L}` / `\p{N}` / `\s` classes and uncased punctuation. The
// contraction-casing golden vectors and the fold split test keep this honest.

export const PRE_TOKENIZE = new RegExp(
  "(?:'s|'t|'re|'ve|'m|'ll|'d)" +
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
    "|\\p{N}" +
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
    "|\\s*[\\r\\n]+" +
    "|\\s+(?!\\S)" +
    "|\\s+",
  "giu",
);

// ---------------------------------------------------------------------------

const SPEECH_TOKEN_RE = /^<\|s_(\d+)\|>$/;

class BpeTokenizer implements Tokenizer {
  private readonly vocab: Map<string, number>;
  private readonly pieces: Map<number, string>;
  private readonly ranks: Map<string, number>;
  /** Added tokens *except* the 12,800 `<|s_n|>` forms. */
  private readonly addedByContent: Map<string, number>;
  private readonly addedById: Map<number, string>;
  /** Alternation over the non-speech added tokens plus one `<|s_n|>` branch. */
  private readonly addedSplitter: RegExp;
  private readonly bpeCache = new Map<string, number[]>();

  constructor(json: TokenizerJson) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.pieces = new Map();
    for (const [piece, id] of this.vocab) this.pieces.set(id, piece);

    this.ranks = new Map();
    json.model.merges.forEach((merge, rank) => {
      const [a, b] = typeof merge === "string" ? splitMergeString(merge) : merge;
      this.ranks.set(`${a} ${b}`, rank);
    });

    this.addedByContent = new Map();
    this.addedById = new Map();
    const named: string[] = [];
    for (const { id, content } of json.added_tokens) {
      this.addedById.set(id, content);
      const speech = SPEECH_TOKEN_RE.exec(content);
      if (speech && speechIndexOf(id) === Number(speech[1])) continue; // regex branch handles it
      this.addedByContent.set(content, id);
      named.push(content);
    }
    // Longest first so that, at a shared start position, the longer literal
    // wins (JS alternation is first-match, not longest-match).
    named.sort((a, b) => b.length - a.length);
    const escaped = named.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.addedSplitter = new RegExp(`${escaped.join("|")}|<\\|s_(\\d+)\\|>`, "g");
  }

  encode(text: string): number[] {
    const normalized = text.normalize("NFC");
    const ids: number[] = [];
    const re = this.addedSplitter;
    re.lastIndex = 0;
    let plainFrom = 0;
    for (let m = re.exec(normalized); m !== null; m = re.exec(normalized)) {
      let id: number | undefined;
      if (m[1] !== undefined) {
        // `<|s_(\d+)\|>` branch: only canonical, in-range forms are added
        // tokens ("<|s_007|>" and "<|s_12800|>" are ordinary text).
        const n = Number(m[1]);
        if (String(n) === m[1] && n < SPEECH_TOKEN_COUNT) id = SPEECH_TOKEN_BASE + n;
      } else {
        id = this.addedByContent.get(m[0]);
      }
      if (id === undefined) {
        // Not actually an added token; rescan from the next character so an
        // added token starting inside the rejected span is still found.
        re.lastIndex = m.index + 1;
        continue;
      }
      this.encodePlain(normalized.slice(plainFrom, m.index), ids);
      ids.push(id);
      plainFrom = m.index + m[0].length;
    }
    this.encodePlain(normalized.slice(plainFrom), ids);
    return ids;
  }

  /** Pre-tokenize + byte-map + BPE one added-token-free span into `ids`. */
  private encodePlain(span: string, ids: number[]): void {
    if (span.length === 0) return;
    for (const [word] of span.matchAll(PRE_TOKENIZE)) {
      const cached = this.bpeCache.get(word);
      if (cached) {
        ids.push(...cached);
        continue;
      }
      const bytes = new TextEncoder().encode(word);
      let parts: string[] = new Array(bytes.length);
      for (let i = 0; i < bytes.length; i += 1) parts[i] = BYTE_TO_CHAR[bytes[i]!]!;
      parts = this.merge(parts);
      const wordIds = parts.map((piece) => {
        const id = this.vocab.get(piece);
        if (id === undefined) {
          // Unreachable with this vocab: all 256 single-byte pieces exist, so
          // every merge result decomposes to known pieces. Guarded anyway —
          // a silent skip here would corrupt the id stream undetectably.
          throw new Error(`piece ${JSON.stringify(piece)} not in vocab`);
        }
        return id;
      });
      this.bpeCache.set(word, wordIds);
      ids.push(...wordIds);
    }
  }

  /** Standard BPE: repeatedly apply the lowest-ranked adjacent merge. */
  private merge(parts: string[]): string[] {
    while (parts.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i < parts.length - 1; i += 1) {
        const rank = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      const merged = parts[bestAt]! + parts[bestAt + 1]!;
      const next: string[] = [];
      for (let i = 0; i < parts.length; i += 1) {
        if (i < parts.length - 1 && parts[i] === parts[bestAt] && parts[i + 1] === parts[bestAt + 1]) {
          next.push(merged);
          i += 1;
        } else {
          next.push(parts[i]!);
        }
      }
      parts = next;
    }
    return parts;
  }

  decode(ids: number[]): string {
    const decoder = new TextDecoder("utf-8", { fatal: false }); // invalid → U+FFFD
    let out = "";
    let pending: number[] = [];
    const flush = () => {
      if (pending.length === 0) return;
      out += decoder.decode(new Uint8Array(pending));
      pending = [];
    };
    for (const id of ids) {
      const speech = speechIndexOf(id);
      if (speech !== null) {
        flush();
        out += `<|s_${speech}|>`;
        continue;
      }
      const added = this.addedById.get(id);
      if (added !== undefined) {
        flush();
        out += added;
        continue;
      }
      const piece = this.pieces.get(id);
      if (piece === undefined) throw new Error(`id ${id} not in vocab`);
      for (const ch of piece) {
        const byte = CHAR_TO_BYTE.get(ch);
        if (byte === undefined) throw new Error(`piece char ${JSON.stringify(ch)} outside byte map`);
        pending.push(byte);
      }
    }
    flush();
    return out;
  }

  encodeChat(userText: string, opts?: EncodeChatOptions): number[] {
    const addGenerationPrompt = opts?.addGenerationPrompt ?? true;
    let prompt = "";
    if (opts?.system !== undefined) {
      prompt += `<|im_start|>system\n${opts.system}<|im_end|>\n`;
    }
    prompt += `<|im_start|>user\n${userText}<|im_end|>\n`;
    if (addGenerationPrompt) prompt += "<|im_start|>assistant\n";
    return this.encode(prompt);
  }
}

/** `"a b"` merge lines: the two pieces never contain a raw space (byte-level). */
function splitMergeString(merge: string): [string, string] {
  const at = merge.indexOf(" ");
  if (at < 0 || merge.indexOf(" ", at + 1) >= 0) {
    throw new Error(`unsplittable merge line ${JSON.stringify(merge)}`);
  }
  return [merge.slice(0, at), merge.slice(at + 1)];
}

export async function loadTokenizer(json: TokenizerJson): Promise<Tokenizer> {
  return new BpeTokenizer(json);
}
