import { InvalidInputError } from "../errors.js";

const HEADER_BYTES = 44;
const BITS_PER_SAMPLE = 16;
const CHANNELS = 1;

/**
 * Encode a mono float signal as a 16 bit PCM RIFF/WAVE file.
 *
 * The result can be handed straight to `new Blob([...], { type: "audio/wav" })`
 * or written to disk in Node.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new InvalidInputError("sampleRate must be a positive finite number.");
  }

  const dataBytes = samples.length * (BITS_PER_SAMPLE / 8);
  const buffer = new ArrayBuffer(HEADER_BYTES + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");

  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, (sampleRate * CHANNELS * BITS_PER_SAMPLE) / 8, true); // byte rate
  view.setUint16(32, (CHANNELS * BITS_PER_SAMPLE) / 8, true); // block align
  view.setUint16(34, BITS_PER_SAMPLE, true);

  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  for (let index = 0; index < samples.length; index += 1) {
    view.setInt16(HEADER_BYTES + index * 2, toInt16(samples[index] as number), true);
  }
  return buffer;
}

function toInt16(sample: number): number {
  const clamped = Math.max(-1, Math.min(1, sample));
  return clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}
