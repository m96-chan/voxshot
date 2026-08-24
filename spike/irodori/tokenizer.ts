/**
 * ModernBERT-ja's tokenizer: SentencePiece Unigram with byte fallback.
 *
 * A different algorithm from the Qwen2 byte-level BPE voxshot already has, and
 * the difference is not cosmetic. **BPE builds one segmentation greedily** by
 * repeatedly applying the lowest-ranked merge. **Unigram scores every
 * segmentation the vocabulary admits and takes the best**, by Viterbi over a
 * lattice — so a piece that BPE would never reach can win here if the pieces
 * around it are cheap enough. Porting one as the other produces plausible ids
 * for common text and different ids for the rest, which is the quiet kind of
 * wrong: the model would simply say something slightly else, fluently.
 *
 * The pipeline, from `tokenizer.json`:
 *
 *   Metaspace(replacement "▁", prepend_scheme "never", split false)
 *     -> Unigram(unk_id 0, byte_fallback true)
 *     -> TemplateProcessing: <s> … </s>
 *
 * There is no normalizer — `normalize_text` in `text.ts` is Irodori's own step
 * and runs before any of this.
 */

export interface TokenizerJson {
  added_tokens: { id: number; content: string; special?: boolean }[];
  model: {
    type: string;
    unk_id?: number;
    byte_fallback?: boolean;
    vocab: [string, number][];
  };
  post_processor?: unknown;
}

export interface Tokenizer {
  /** Ids for one string, wrapped in `<s>` / `</s>` as the template does. */
  encode(text: string): number[];
  /** Ids without the template, for checking the Unigram half on its own. */
  encodePieces(text: string): number[];
  decode(ids: number[]): string;
  readonly bosId: number;
  readonly eosId: number;
  readonly unkId: number;
  readonly vocabSize: number;
}

/** Metaspace's replacement character, U+2581. */
const METASPACE = "▁";

/**
 * What the lattice scores an uncoverable character at.
 *
 * `unk_score = min(vocab scores) - K` with K = 10, so an unknown piece is worse
 * than any real one by a wide margin but stays finite.
 *
 * **Byte fallback does not compete in the lattice**, and getting that wrong is
 * the first thing this port did. The `<0xNN>` pieces carry a score of **0** in
 * this vocabulary while every real piece is negative — so offering them as
 * candidates makes byte fallback beat every word, every time, and the whole
 * corpus tokenizes one byte at a time. It is applied *after* the path is
 * chosen, to characters the vocabulary could not cover at all.
 */
const UNK_PENALTY = 10.0;

class UnigramTokenizer implements Tokenizer {
  readonly bosId: number;
  readonly eosId: number;
  readonly unkId: number;
  readonly vocabSize: number;

  /** piece -> [id, score]. */
  private readonly pieces = new Map<string, [number, number]>();
  private readonly byId: string[];
  /** Longest piece in code units, to bound the lattice's inner loop. */
  private readonly maxPieceLength: number;
  private readonly unkScore: number;
  private readonly byteFallback: boolean;
  /** `<0xNN>` ids, indexed by byte, when byte fallback is available. */
  private readonly byteTokens: (number | undefined)[];
  /** Added tokens are matched literally before the lattice runs. */
  private readonly addedSplitter: RegExp | null;
  private readonly addedByContent = new Map<string, number>();

  constructor(json: TokenizerJson) {
    if (json.model.type !== "Unigram") {
      throw new Error(`expected a Unigram model, got ${json.model.type}`);
    }
    this.vocabSize = json.model.vocab.length;
    this.byId = new Array<string>(this.vocabSize);
    let longest = 0;
    let worst = Infinity;
    json.model.vocab.forEach(([piece, score], id) => {
      // First occurrence wins: a duplicate piece later in the list is
      // unreachable in the reference too, since it builds the same map.
      if (!this.pieces.has(piece)) this.pieces.set(piece, [id, score]);
      this.byId[id] = piece;
      if (piece.length > longest) longest = piece.length;
      if (score < worst) worst = score;
    });
    this.maxPieceLength = longest;
    this.unkScore = worst - UNK_PENALTY;
    this.unkId = json.model.unk_id ?? 0;
    this.byteFallback = json.model.byte_fallback === true;

    this.byteTokens = new Array<number | undefined>(256);
    if (this.byteFallback) {
      for (let byte = 0; byte < 256; byte += 1) {
        const name = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
        this.byteTokens[byte] = this.pieces.get(name)?.[0];
      }
    }

    const named: string[] = [];
    for (const { id, content } of json.added_tokens) {
      this.addedByContent.set(content, id);
      named.push(content);
    }
    // Longest first: JS alternation is first-match, not longest-match, so
    // `<|tool_calls|>` has to be tried before any shorter prefix of it.
    named.sort((a, b) => b.length - a.length);
    this.addedSplitter =
      named.length > 0
        ? new RegExp(named.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g")
        : null;

    const bos = this.addedByContent.get("<s>");
    const eos = this.addedByContent.get("</s>");
    if (bos === undefined || eos === undefined) {
      throw new Error("tokenizer.json has no <s>/</s> in added_tokens");
    }
    this.bosId = bos;
    this.eosId = eos;
  }

  encode(text: string): number[] {
    return [this.bosId, ...this.encodePieces(text), this.eosId];
  }

  encodePieces(text: string): number[] {
    const ids: number[] = [];
    if (this.addedSplitter) {
      const re = this.addedSplitter;
      re.lastIndex = 0;
      let from = 0;
      for (let m = re.exec(text); m !== null; m = re.exec(text)) {
        const id = this.addedByContent.get(m[0]);
        if (id === undefined) {
          re.lastIndex = m.index + 1;
          continue;
        }
        this.encodeSpan(text.slice(from, m.index), ids);
        ids.push(id);
        from = m.index + m[0].length;
      }
      this.encodeSpan(text.slice(from), ids);
    } else {
      this.encodeSpan(text, ids);
    }
    return ids;
  }

  /**
   * Metaspace, then Viterbi over the lattice.
   *
   * `prepend_scheme: "never"` and `split: false`, so Metaspace is exactly a
   * substitution of spaces — no leading marker, no splitting into words. The
   * whole span goes into one lattice.
   */
  private encodeSpan(span: string, into: number[]): void {
    if (span.length === 0) return;
    const text = span.split(" ").join(METASPACE);
    const n = text.length;

    // best[i] = score of the best path covering text[0..i)
    const best = new Float64Array(n + 1).fill(-Infinity);
    const backId = new Int32Array(n + 1).fill(-1);
    const backFrom = new Int32Array(n + 1).fill(-1);
    // Byte-fallback steps carry several ids for one character, so they are
    // remembered separately rather than squeezed into `backId`.
    const backBytes: (number[] | null)[] = new Array(n + 1).fill(null);
    best[0] = 0;

    for (let start = 0; start < n; start += 1) {
      if (best[start] === -Infinity) continue;
      const limit = Math.min(n, start + this.maxPieceLength);
      let matched = false;
      for (let end = start + 1; end <= limit; end += 1) {
        const entry = this.pieces.get(text.slice(start, end));
        if (!entry) continue;
        matched = true;
        const score = best[start]! + entry[1];
        if (score > best[end]!) {
          best[end] = score;
          backId[end] = entry[0];
          backFrom[end] = start;
          backBytes[end] = null;
        }
      }

      // Only what the vocabulary could not cover, and one whole code point at
      // a time — splitting a surrogate pair would encode to different bytes
      // than the character it came from.
      if (!matched) {
        const codePoint = text.codePointAt(start)!;
        const end = start + (codePoint > 0xffff ? 2 : 1);
        const score = best[start]! + this.unkScore;
        if (score > best[end]!) {
          best[end] = score;
          backId[end] = -1;
          backFrom[end] = start;
          backBytes[end] = this.expandUnknown(text.slice(start, end));
        }
      }
    }

    if (best[n] === -Infinity) {
      throw new Error(`no path through the lattice for ${JSON.stringify(span)}`);
    }

    const reversed: number[] = [];
    for (let at = n; at > 0; ) {
      const from = backFrom[at]!;
      const bytes = backBytes[at];
      if (bytes) reversed.push(...[...bytes].reverse());
      else reversed.push(backId[at]!);
      at = from;
    }
    for (let i = reversed.length - 1; i >= 0; i -= 1) into.push(reversed[i]!);
  }

  /**
   * An uncoverable character, as the ids that stand in for it.
   *
   * Byte pieces when the vocabulary has them, the unk token otherwise. This
   * runs after the path is chosen, so it contributes no score — see
   * {@link UNK_PENALTY}.
   */
  private expandUnknown(chunk: string): number[] {
    if (!this.byteFallback) return [this.unkId];
    const ids: number[] = [];
    for (const byte of new TextEncoder().encode(chunk)) {
      const id = this.byteTokens[byte];
      if (id === undefined) return [this.unkId];
      ids.push(id);
    }
    return ids;
  }

  decode(ids: number[]): string {
    // The decoder is Replace(▁ -> " ") then ByteFallback then Fuse: byte
    // pieces are gathered back into bytes and decoded together, so a
    // multi-byte character split across pieces comes back whole.
    let out = "";
    let pending: number[] = [];
    const flush = () => {
      if (pending.length === 0) return;
      out += new TextDecoder().decode(new Uint8Array(pending));
      pending = [];
    };
    for (const id of ids) {
      if (id === this.bosId || id === this.eosId) continue;
      const piece = this.byId[id];
      if (piece === undefined) throw new Error(`id ${id} not in vocab`);
      const byteMatch = /^<0x([0-9A-F]{2})>$/.exec(piece);
      if (byteMatch) {
        pending.push(parseInt(byteMatch[1]!, 16));
        continue;
      }
      flush();
      out += piece.split(METASPACE).join(" ");
    }
    flush();
    return out;
  }
}

export function loadTokenizer(json: TokenizerJson): Tokenizer {
  return new UnigramTokenizer(json);
}
