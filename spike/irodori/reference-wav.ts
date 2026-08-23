import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * The reference's own output for the text this port renders, as a WAV.
 *
 *     cd spike/irodori && npx tsx reference-wav.ts
 *
 * `dump_golden.py` records the pipeline's final waveform, and this writes it
 * through the *same* encoder `synthesize.ts` uses. So a difference between
 * `samples/reference-01.wav` and `samples/port-01.wav` is a difference between
 * torch and this port, not between two WAV writers — and if the port's output
 * were silent or clipped, this file says whether the writer could be the cause.
 *
 * They will not be sample-identical. `torch.randn` with a seeded generator is
 * not reproducible in JavaScript, so the two are different draws from the same
 * distribution along the same conditioning.
 */

const HERE = dirname(fileURLToPath(import.meta.url));
const SAMPLE_RATE = 48000;

const bytes = readFileSync(join(HERE, "golden", "audio.f32"));
const copy = new Uint8Array(bytes.byteLength);
copy.set(bytes);
const samples = new Float32Array(copy.buffer);

const data = Buffer.alloc(samples.length * 2);
let peak = 0;
let energy = 0;
for (let index = 0; index < samples.length; index += 1) {
  const value = samples[index]!;
  peak = Math.max(peak, Math.abs(value));
  energy += value * value;
  data.writeInt16LE(Math.round(Math.max(-1, Math.min(1, value)) * 32767), index * 2);
}
const header = Buffer.alloc(44);
header.write("RIFF", 0);
header.writeUInt32LE(36 + data.length, 4);
header.write("WAVEfmt ", 8);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20);
header.writeUInt16LE(1, 22);
header.writeUInt32LE(SAMPLE_RATE, 24);
header.writeUInt32LE(SAMPLE_RATE * 2, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write("data", 36);
header.writeUInt32LE(data.length, 40);

const output = join(HERE, "samples", "reference-01.wav");
writeFileSync(output, Buffer.concat([header, data]));
console.log(
  `wrote ${output}\n  ${samples.length} samples (${(samples.length / SAMPLE_RATE).toFixed(2)}s), ` +
    `peak ${peak.toFixed(3)}, rms ${Math.sqrt(energy / samples.length).toFixed(3)}`,
);
