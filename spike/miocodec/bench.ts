import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { decode, MIOCODEC_24K, Weights } from "./decoder.js";
import { Safetensors } from "./safetensors.js";

const fixture = JSON.parse(readFileSync("../../examples/mio-tts-fixture.json", "utf8"));
const hub = join(homedir(), ".cache", "huggingface", "hub", "models--Aratako--MioCodec-25Hz-24kHz");
const rev = readFileSync(join(hub, "refs", "main"), "utf8").trim();
const bytes = readFileSync(join(hub, "snapshots", rev, "model.safetensors"));
const weights = new Weights(Safetensors.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer));

const t0 = performance.now();
const { waveform } = await decode(Float32Array.from(fixture.tokens), Float32Array.from(fixture.global_embedding), fixture.stft_length, MIOCODEC_24K, weights);
const ms = performance.now() - t0;
const seconds = waveform.length / fixture.sample_rate;
console.log(`${fixture.tokens.length} tokens -> ${fixture.stft_length} frames -> ${waveform.length} samples (${seconds.toFixed(2)} s)`);
console.log(`decode ${ms.toFixed(0)} ms, RTF ${(ms / 1000 / seconds).toFixed(2)}`);
const ref = new Float32Array(readFileSync("golden/demo_reference.f32").buffer);
let worst = 0, scale = 0;
for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]!));
for (let i = 0; i < Math.min(ref.length, waveform.length); i++) worst = Math.max(worst, Math.abs(waveform[i]! - ref[i]!));
console.log(`vs reference decode: ${ref.length} samples, worst abs ${worst.toExponential(3)}, rel ${(worst/scale).toExponential(3)}`);
