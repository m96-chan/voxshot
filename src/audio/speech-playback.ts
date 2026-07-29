import type { StreamingPlayback } from "../platform.js";

/** A running utterance started by `ZeroVox.play()`. */
export interface SpeechPlayback {
  /** Resolves when playback finished or was stopped; rejects on error. */
  readonly done: Promise<void>;
  /** Stop playing and discard everything, including unsynthesized chunks. */
  stop(): Promise<void>;
  /** Jump past the chunk that is currently audible. */
  skip(): Promise<void>;
  /** Playback gain. Applied immediately, also mid-utterance. */
  setVolume(volume: number): void;
}

export interface SpeechPlaybackInput {
  /** Text chunks in speaking order. */
  chunks: readonly string[];
  /** Render one chunk. Called with a one-chunk lookahead. */
  synthesize(chunk: string): Promise<Float32Array>;
  /** Open the output stream. Called once, lazily. */
  open(): Promise<StreamingPlayback>;
  /** Initial gain. */
  volume?: number;
}

/** What the device currently holds of one chunk, in stream sample offsets. */
interface WrittenChunk {
  index: number;
  samples: Float32Array;
  start: number;
  end: number;
}

/**
 * Drive a chunked utterance through a {@link StreamingPlayback}.
 *
 * While chunk *n* is playing, chunk *n + 1* is being synthesized — exactly one
 * chunk of lookahead, so stopping wastes at most one render and memory stays
 * bounded. Device backpressure comes from `write()`, which resolves when the
 * stream wants the next chunk.
 */
export function startSpeechPlayback(input: SpeechPlaybackInput): SpeechPlayback {
  return new SpeechPlaybackController(input);
}

class SpeechPlaybackController implements SpeechPlayback {
  readonly done: Promise<void>;

  readonly #input: SpeechPlaybackInput;
  #playback: StreamingPlayback | undefined;
  #pendingVolume: number | undefined;

  /** Chunks handed to the device that may not have fully played yet. */
  #written: WrittenChunk[] = [];
  /** Total samples handed to the device and not discarded by a flush. */
  #writtenTotal = 0;
  /** Write issued by {@link skip} that the main loop must await. */
  #rewrite: Promise<void> = Promise.resolve();
  /** Bumped by every flush, so writes resolved *by* a flush do not prune. */
  #flushGeneration = 0;

  #stopped = false;
  #finished = false;

  constructor(input: SpeechPlaybackInput) {
    this.#input = input;
    this.#pendingVolume = input.volume;
    this.done = this.#run();
  }

  async stop(): Promise<void> {
    if (this.#stopped || this.#finished) {
      return;
    }
    this.#stopped = true;
    await this.#playback?.stop();
  }

  async skip(): Promise<void> {
    const playback = this.#playback;
    if (!playback || this.#stopped || this.#finished) {
      return;
    }

    this.#flushGeneration += 1;
    const played = await playback.flush();
    this.#writtenTotal = played;

    // Everything that already played is history; the first remaining chunk is
    // the one that was audible — that is the one being skipped.
    const remaining = this.#written.filter((chunk) => chunk.end > played);
    const survivors = remaining.slice(1);
    this.#written = [];

    // Re-enqueue chunks the flush discarded before they became audible. The
    // writes start now (preserving order ahead of the main loop's next chunk)
    // but are not awaited here: their resolution is the device asking for
    // more audio, which the main loop waits for via #rewrite.
    this.#rewrite = (async () => {
      for (const chunk of survivors) {
        if (this.#stopped) {
          return;
        }
        await this.#write(chunk.index, chunk.samples);
      }
    })();
  }

  setVolume(volume: number): void {
    if (this.#playback) {
      this.#playback.setVolume(volume);
    } else {
      this.#pendingVolume = volume;
    }
  }

  async #run(): Promise<void> {
    const { chunks, synthesize } = this.#input;
    try {
      const playback = await this.#input.open();
      this.#playback = playback;
      // stop() may have been called while the device was still opening.
      if (this.#stopped) {
        await playback.stop();
        return;
      }
      if (this.#pendingVolume !== undefined) {
        playback.setVolume(this.#pendingVolume);
        this.#pendingVolume = undefined;
      }

      let next: Promise<Float32Array> | undefined =
        chunks.length > 0 ? synthesize(chunks[0] as string) : undefined;

      for (let index = 0; index < chunks.length; index += 1) {
        const samples = (await next) as Float32Array;
        if (this.#stopped) {
          return;
        }
        next = index + 1 < chunks.length ? synthesize(chunks[index + 1] as string) : undefined;

        await this.#rewrite;
        if (this.#stopped) {
          return;
        }
        await this.#write(index, samples);
        if (this.#stopped) {
          return;
        }
      }

      await this.#rewrite;
      if (!this.#stopped) {
        this.#finished = true;
        await playback.end();
      }
    } catch (cause) {
      this.#stopped = true;
      await this.#playback?.stop().catch(() => undefined);
      throw cause;
    }
  }

  async #write(index: number, samples: Float32Array): Promise<void> {
    const playback = this.#playback as StreamingPlayback;
    const record: WrittenChunk = {
      index,
      samples,
      start: this.#writtenTotal,
      end: this.#writtenTotal + samples.length,
    };
    this.#writtenTotal = record.end;
    this.#written.push(record);

    const generation = this.#flushGeneration;
    await playback.write(samples);

    // A write that resolves *normally* means the device drained below its
    // low-water mark, so every earlier chunk has finished playing and can
    // never be flushed back — drop the records. A write resolved by a flush
    // proves nothing; skip() has already rebuilt the records in that case.
    if (generation === this.#flushGeneration) {
      this.#written = this.#written.filter((chunk) => chunk.end >= record.end);
    }
  }
}
