// ../../../web-xpu-ops/ops/activation/reference.ts
var ACTIVATION = {
  relu2: 0,
  silu: 1,
  elu: 2,
  tanh: 3,
  gelu: 4,
  gelu_tanh: 5
};
function erf(x) {
  if (x < 0) return -erf(-x);
  if (x >= 6) return 1;
  const twoXSquared = 2 * x * x;
  let term = 1;
  let sum = 1;
  for (let n = 1; n < 400; n += 1) {
    term *= twoXSquared / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return 2 * x / Math.sqrt(Math.PI) * Math.exp(-x * x) * sum;
}
function activation({ input, kind, alpha = 1 }) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i];
    output[i] = apply(x, kind, alpha);
  }
  return output;
}
function apply(x, kind, alpha) {
  switch (kind) {
    case ACTIVATION.relu2:
      return Math.max(0, x) ** 2;
    case ACTIVATION.silu:
      return x / (1 + Math.exp(-x));
    case ACTIVATION.elu:
      return x > 0 ? x : alpha * (Math.exp(x) - 1);
    case ACTIVATION.tanh:
      return Math.tanh(x);
    case ACTIVATION.gelu:
      return 0.5 * x * (1 + erf(x / Math.SQRT2));
    default:
      return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  }
}

// ../../../web-xpu-ops/ops/attention/reference.ts
function defaultScale(D) {
  return 1 / Math.sqrt(D);
}
function resolveMask(args, op) {
  const { mask, B, H, L, S } = args;
  const shape = args.maskShape ?? [B, 1, 1];
  if (!mask) {
    if (args.maskShape) throw new Error(`${op}(): maskShape given without a mask`);
    return { shape, at: () => 0 };
  }
  if (args.causal) {
    throw new Error(`${op}(): causal and mask are exclusive, as torch rejects is_causal with attn_mask`);
  }
  const [mb, mh, mr] = shape;
  const full = { maskBatch: [mb, B], maskHeads: [mh, H], maskRows: [mr, L] };
  for (const [name, [got, want]] of Object.entries(full)) {
    if (got !== 1 && got !== want) {
      throw new Error(`${op}(): ${name} must be 1 or ${want}, got ${got}`);
    }
  }
  const expected = mb * mh * mr * S;
  if (mask.length !== expected) {
    throw new Error(`${op}(): mask ${mb}x${mh}x${mr}x${S} needs ${expected} elements, got ${mask.length}`);
  }
  return {
    shape,
    at: (b, h, i, j) => mask[(((mb === 1 ? 0 : b) * mh + (mh === 1 ? 0 : h)) * mr + (mr === 1 ? 0 : i)) * S + j]
  };
}
function attention(args) {
  const { q, k, v, B, H, L, S, D, Dv } = args;
  const causal = args.causal ?? false;
  const queryOffset = args.queryOffset ?? 0;
  const scale = args.scale ?? defaultScale(D);
  const { at: bias } = resolveMask(args, "attention");
  const probs = new Float32Array(B * H * L * S);
  const output = new Float32Array(B * H * L * Dv);
  for (let b = 0; b < B; b += 1) {
    for (let h = 0; h < H; h += 1) {
      const head = b * H + h;
      const qHead = head * L * D;
      const kHead = head * S * D;
      const vHead = head * S * Dv;
      const pHead = head * L * S;
      const oHead = head * L * Dv;
      for (let i = 0; i < L; i += 1) {
        const row = new Float64Array(S);
        for (let j = 0; j < S; j += 1) {
          if (causal && j > i + queryOffset) {
            row[j] = -Infinity;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < D; d += 1) dot += q[qHead + i * D + d] * k[kHead + j * D + d];
          row[j] = dot * scale + bias(b, h, i, j);
        }
        let max = -Infinity;
        for (let j = 0; j < S; j += 1) max = Math.max(max, row[j]);
        if (max === -Infinity) continue;
        let sum = 0;
        for (let j = 0; j < S; j += 1) sum += Math.exp(row[j] - max);
        for (let j = 0; j < S; j += 1) probs[pHead + i * S + j] = Math.exp(row[j] - max) / sum;
        for (let c = 0; c < Dv; c += 1) {
          let acc = 0;
          for (let j = 0; j < S; j += 1) acc += probs[pHead + i * S + j] * v[vHead + j * Dv + c];
          output[oHead + i * Dv + c] = acc;
        }
      }
    }
  }
  return { probs, output };
}

// ../../../web-xpu-ops/ops/conv/reference.ts
function conv1dOutputLength({
  L,
  K,
  stride = 1,
  padding: padding2 = 0,
  dilation = 1
}) {
  return Math.floor((L + 2 * padding2 - dilation * (K - 1) - 1) / stride) + 1;
}
function conv1d({
  input,
  weight,
  bias,
  N,
  Cin,
  Cout,
  L,
  K,
  stride = 1,
  padding: padding2 = 0,
  dilation = 1,
  groups = 1
}) {
  if (Cin % groups !== 0 || Cout % groups !== 0) {
    throw new Error(`conv1d(): Cin=${Cin} and Cout=${Cout} must both be divisible by groups=${groups}`);
  }
  const Lout = conv1dOutputLength({ L, K, stride, padding: padding2, dilation });
  if (Lout <= 0) {
    throw new Error(`conv1d(): kernel size ${K} (dilated ${dilation * (K - 1) + 1}) exceeds padded input size ${L + 2 * padding2}`);
  }
  if (input.length !== N * Cin * L) {
    throw new Error(`conv1d(): expected ${N * Cin * L} input elements, got ${input.length}`);
  }
  if (weight.length !== Cout * (Cin / groups) * K) {
    throw new Error(`conv1d(): expected ${Cout * (Cin / groups) * K} weight elements, got ${weight.length}`);
  }
  const inPerGroup = Cin / groups;
  const outPerGroup = Cout / groups;
  const output = new Float32Array(N * Cout * Lout);
  for (let n = 0; n < N; n += 1) {
    for (let oc = 0; oc < Cout; oc += 1) {
      const group = Math.floor(oc / outPerGroup);
      for (let ol = 0; ol < Lout; ol += 1) {
        let acc = bias ? bias[oc] : 0;
        for (let icLocal = 0; icLocal < inPerGroup; icLocal += 1) {
          const ic = group * inPerGroup + icLocal;
          for (let k = 0; k < K; k += 1) {
            const il = ol * stride + k * dilation - padding2;
            if (il < 0 || il >= L) continue;
            acc += input[(n * Cin + ic) * L + il] * weight[(oc * inPerGroup + icLocal) * K + k];
          }
        }
        output[(n * Cout + oc) * Lout + ol] = acc;
      }
    }
  }
  return output;
}

// ../../../web-xpu-ops/ops/conv_transpose/reference.ts
function convTranspose1dOutputLength({
  L,
  K,
  stride = 1,
  padding: padding2 = 0,
  outputPadding = 0,
  dilation = 1
}) {
  return (L - 1) * stride - 2 * padding2 + dilation * (K - 1) + outputPadding + 1;
}
function convTranspose1d({
  input,
  weight,
  bias,
  N,
  Cin,
  Cout,
  L,
  K,
  stride = 1,
  padding: padding2 = 0,
  outputPadding = 0,
  dilation = 1,
  groups = 1
}) {
  if (padding2 < 0) {
    throw new Error(`convTranspose1d(): negative padding is not supported, got padding=${padding2}`);
  }
  if (outputPadding < 0 || outputPadding >= stride && outputPadding >= dilation) {
    throw new Error(
      `convTranspose1d(): output_padding=${outputPadding} must be smaller than either stride=${stride} or dilation=${dilation}`
    );
  }
  if (Cin % groups !== 0 || Cout % groups !== 0) {
    throw new Error(`convTranspose1d(): Cin=${Cin} and Cout=${Cout} must both be divisible by groups=${groups}`);
  }
  const Lout = convTranspose1dOutputLength({ L, K, stride, padding: padding2, outputPadding, dilation });
  if (Lout <= 0) {
    throw new Error(`convTranspose1d(): output size is too small (${Lout}); padding=${padding2} crops more than the convolution produces`);
  }
  if (input.length !== N * Cin * L) {
    throw new Error(`convTranspose1d(): expected ${N * Cin * L} input elements, got ${input.length}`);
  }
  if (weight.length !== Cin * (Cout / groups) * K) {
    throw new Error(`convTranspose1d(): expected ${Cin * (Cout / groups) * K} weight elements, got ${weight.length}`);
  }
  const inPerGroup = Cin / groups;
  const outPerGroup = Cout / groups;
  const output = new Float32Array(N * Cout * Lout);
  if (bias) {
    for (let n = 0; n < N; n += 1) {
      for (let oc = 0; oc < Cout; oc += 1) {
        for (let ol = 0; ol < Lout; ol += 1) {
          output[(n * Cout + oc) * Lout + ol] = bias[oc];
        }
      }
    }
  }
  for (let n = 0; n < N; n += 1) {
    for (let ic = 0; ic < Cin; ic += 1) {
      const group = Math.floor(ic / inPerGroup);
      for (let ocLocal = 0; ocLocal < outPerGroup; ocLocal += 1) {
        const oc = group * outPerGroup + ocLocal;
        for (let l = 0; l < L; l += 1) {
          const x = input[(n * Cin + ic) * L + l];
          for (let k = 0; k < K; k += 1) {
            const ol = l * stride + k * dilation - padding2;
            if (ol < 0 || ol >= Lout) continue;
            output[(n * Cout + oc) * Lout + ol] += x * weight[(ic * outPerGroup + ocLocal) * K + k];
          }
        }
      }
    }
  }
  return output;
}

// ../../../web-xpu-ops/ops/group_norm/reference.ts
function groupNorm({ input, weight, bias, N, C, L, G, eps }) {
  if (G <= 0 || C % G !== 0) {
    throw new Error(
      `group_norm: expected number of channels (${C}) to be divisible by num_groups (${G})`
    );
  }
  const output = new Float32Array(N * C * L);
  const channelsPerGroup = C / G;
  const count = channelsPerGroup * L;
  for (let n = 0; n < N; n += 1) {
    for (let g = 0; g < G; g += 1) {
      const start = (n * C + g * channelsPerGroup) * L;
      let sum = 0;
      for (let i = 0; i < count; i += 1) {
        sum += input[start + i];
      }
      const mean = sum / count;
      let sumSquaredDeviations = 0;
      for (let i = 0; i < count; i += 1) {
        const deviation = input[start + i] - mean;
        sumSquaredDeviations += deviation * deviation;
      }
      const variance = sumSquaredDeviations / count;
      const scale = 1 / Math.sqrt(variance + eps);
      for (let i = 0; i < count; i += 1) {
        const channel = g * channelsPerGroup + Math.floor(i / L);
        output[start + i] = (input[start + i] - mean) * scale * weight[channel] + bias[channel];
      }
    }
  }
  return output;
}

// ../../../web-xpu-ops/ops/layernorm/reference.ts
function layernorm({ input, weight, bias, N, D, eps }) {
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row += 1) {
    let sum = 0;
    for (let col = 0; col < D; col += 1) {
      sum += input[row * D + col];
    }
    const mean = sum / D;
    let sumSquaredDeviations = 0;
    for (let col = 0; col < D; col += 1) {
      const deviation = input[row * D + col] - mean;
      sumSquaredDeviations += deviation * deviation;
    }
    const variance = sumSquaredDeviations / D;
    const scale = 1 / Math.sqrt(variance + eps);
    for (let col = 0; col < D; col += 1) {
      output[row * D + col] = (input[row * D + col] - mean) * scale * weight[col] + bias[col];
    }
  }
  return output;
}

// ../../../web-xpu-ops/ops/matmul/reference.ts
function matmul({ a, b, M, N, K }) {
  const output = new Float32Array(M * N);
  for (let row = 0; row < M; row += 1) {
    for (let col = 0; col < N; col += 1) {
      let sum = 0;
      for (let k = 0; k < K; k += 1) {
        sum += a[row * K + k] * b[k * N + col];
      }
      output[row * N + col] = sum;
    }
  }
  return output;
}

// ../../../web-xpu-ops/ops/rope/reference.ts
var UNSCALED = {
  interpolationFactor: 1,
  rampLow: 0,
  rampHigh: 1,
  attentionFactor: 1
};
function ropeFrequencyParams(headDim, thetaBase, scaling) {
  if (!scaling) return { ...UNSCALED, effectiveBase: thetaBase };
  if (scaling.kind === "ntk") {
    return {
      ...UNSCALED,
      effectiveBase: thetaBase * Math.pow(scaling.factor, headDim / (headDim - 2))
    };
  }
  const { factor, originalContextLength, betaFast = 32, betaSlow = 1 } = scaling;
  const correctionDim = (rotations) => headDim * Math.log(originalContextLength / (rotations * 2 * Math.PI)) / (2 * Math.log(thetaBase));
  const rampLow = Math.max(Math.floor(correctionDim(betaFast)), 0);
  let rampHigh = Math.min(Math.ceil(correctionDim(betaSlow)), headDim - 1);
  if (rampHigh === rampLow) rampHigh += 1e-3;
  return {
    effectiveBase: thetaBase,
    interpolationFactor: factor,
    rampLow,
    rampHigh,
    // §3.4: √(1/t) = 0.1·ln(s) + 1, and 1 when there is nothing to extend.
    attentionFactor: scaling.attentionFactor ?? (factor <= 1 ? 1 : 0.1 * Math.log(factor) + 1)
  };
}
function invFreq({ effectiveBase, interpolationFactor, rampLow, rampHigh }, headDim, pair) {
  const extrapolation = Math.pow(effectiveBase, -2 * pair / headDim);
  const interpolation = extrapolation / interpolationFactor;
  const ramp = Math.min(Math.max((pair - rampLow) / (rampHigh - rampLow), 0), 1);
  return extrapolation + (interpolation - extrapolation) * ramp;
}
function rope({
  input,
  N,
  numHeads,
  headDim,
  posOffset,
  thetaBase,
  scaling,
  cache,
  headOffset = 0,
  headCount = numHeads
}) {
  const output = new Float32Array(input.length);
  const halfDim = headDim / 2;
  const freq = ropeFrequencyParams(headDim, thetaBase, scaling);
  const { attentionFactor } = freq;
  if (cache) {
    if (cache.headDim !== headDim) {
      throw new Error(`rope: cache holds headDim ${cache.headDim}, called with ${headDim}`);
    }
    for (const key of Object.keys(freq)) {
      if (cache.freq[key] !== freq[key]) {
        throw new Error(
          `rope: cache was built with ${key}=${cache.freq[key]}, called with ${key}=${freq[key]}`
        );
      }
    }
  }
  if (headOffset < 0 || headCount < 0 || headOffset + headCount > numHeads) {
    throw new Error(
      `rope: head range [${headOffset}, ${headOffset + headCount}) does not fit ${numHeads} heads`
    );
  }
  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      if (head < headOffset || head >= headOffset + headCount) {
        const from = (token * numHeads + head) * headDim;
        for (let i = 0; i < headDim; i += 1) output[from + i] = input[from + i];
        continue;
      }
      for (let pair = 0; pair < halfDim; pair += 1) {
        const pos = token + posOffset;
        let cos;
        let sin;
        if (cache && pos < cache.positions) {
          const at = (pos * halfDim + pair) * 2;
          cos = cache.table[at];
          sin = cache.table[at + 1];
        } else {
          const theta = pos * invFreq(freq, headDim, pair);
          cos = Math.cos(theta) * attentionFactor;
          sin = Math.sin(theta) * attentionFactor;
        }
        const base = (token * numHeads + head) * headDim + pair * 2;
        const x0 = input[base];
        const x1 = input[base + 1];
        output[base] = x0 * cos - x1 * sin;
        output[base + 1] = x0 * sin + x1 * cos;
      }
    }
  }
  return output;
}

// ../../../web-xpu-ops/ops/stft/reference.ts
function stftBins(nFft) {
  return Math.floor(nFft / 2) + 1;
}
function istftLength(nFft, hop, frames, mode = true) {
  return nFft + hop * (frames - 1) - 2 * padding(nFft, hop, mode);
}
function istftMaxLength(nFft, hop, frames, mode = true) {
  const trim = padding(nFft, hop, mode);
  return resolve(mode) === "same" ? nFft + hop * (frames - 1) - 2 * trim : nFft + hop * (frames - 1) - trim;
}
function resolve(mode) {
  if (mode === true) return "center";
  if (mode === false) return "none";
  return mode;
}
function padding(nFft, hop, mode) {
  const resolved = resolve(mode);
  if (resolved === "none") return 0;
  if (resolved === "same") return Math.floor((nFft - hop) / 2);
  return Math.floor(nFft / 2);
}
function hannWindow(n, periodic = true) {
  const denominator = periodic ? n : n - 1;
  return Float32Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / denominator));
}
function istft({
  real,
  imag,
  frames,
  nFft,
  hop,
  window,
  center,
  padding: mode,
  length
}) {
  if (center !== void 0 && mode !== void 0) {
    throw new Error("give either center or padding, not both");
  }
  const resolved = resolve(mode ?? center ?? true);
  if (nFft < 1) throw new Error(`nFft must be positive, got ${nFft}`);
  if (hop < 1) throw new Error(`hop must be positive, got ${hop}`);
  if (window.length !== nFft) throw new Error(`window must be ${nFft} long, got ${window.length}`);
  const bins = stftBins(nFft);
  for (const [name, side] of [["real", real], ["imag", imag]]) {
    if (side.length !== frames * bins) {
      throw new Error(`${name} must be ${frames} x ${bins} = ${frames * bins}, got ${side.length}`);
    }
  }
  const samples = length ?? istftLength(nFft, hop, frames, resolved);
  const reach = istftMaxLength(nFft, hop, frames, resolved);
  if (samples > reach) {
    throw new Error(`${frames} frames reach ${reach} samples, cannot produce ${samples}`);
  }
  const pad = padding(nFft, hop, resolved);
  const even = nFft % 2 === 0;
  const output = new Float32Array(samples);
  for (let t = 0; t < samples; t += 1) {
    const position = t + pad;
    let numerator = 0;
    let envelope = 0;
    for (let frame = 0; frame < frames; frame += 1) {
      const n = position - frame * hop;
      if (n < 0 || n >= nFft) continue;
      const base = frame * bins;
      let acc = real[base];
      for (let k = 1; k < bins; k += 1) {
        const weight = even && k === bins - 1 ? 1 : 2;
        const angle = 2 * Math.PI * (k * n % nFft) / nFft;
        acc += weight * (real[base + k] * Math.cos(angle) - imag[base + k] * Math.sin(angle));
      }
      const w = window[n];
      numerator += w * (acc / nFft);
      envelope += w * w;
    }
    if (envelope < NOLA_FLOOR) {
      throw new Error(
        `window fails NOLA at sample ${t}: the w\xB2 envelope is ${envelope}, below ${NOLA_FLOOR}`
      );
    }
    output[t] = numerator / envelope;
  }
  return output;
}
var NOLA_FLOOR = 1e-11;

// decoder.ts
var cpuBackend = {
  name: "reference (CPU)",
  async matmul(a, b, M, N, K) {
    return matmul({ a, b, M, N, K });
  },
  async conv1d(input, weight, bias, Cin, Cout, L, K, padding2) {
    return conv1d({ input, weight, bias: bias ?? void 0, N: 1, Cin, Cout, L, K, padding: padding2 });
  },
  async istft(real, imag, window, frames, nFft, hop) {
    return istft({ real, imag, frames, nFft, hop, window, padding: "same" });
  }
};
var NORM_EPS = 1e-5;
var GROUP_NORM_EPS = 1e-6;
var MIOCODEC_24K = {
  nFft: 1920,
  hopLength: 480,
  sampleRate: 24e3,
  waveResnetNumBlocks: 2,
  waveResnetKernelSize: 3,
  waveResnetNumGroups: 32,
  fsqLevels: [8, 8, 8, 5, 5],
  prenet: { dim: 768, layers: 6, heads: 12, windowSize: 65, ropeTheta: 1e4, outputDim: 512 },
  decoder: {
    dim: 512,
    layers: 8,
    heads: 8,
    windowSize: 65,
    ropeTheta: 1e4,
    adaLnConditionDim: 128
  }
};
var transposed = /* @__PURE__ */ new WeakMap();
async function linear(x, weight, bias, backend) {
  const [outFeatures, inFeatures] = weight.shape;
  const rows = x.data.length / inFeatures;
  let b = transposed.get(weight.data);
  if (!b) {
    b = new Float32Array(inFeatures * outFeatures);
    for (let o = 0; o < outFeatures; o += 1) {
      for (let i = 0; i < inFeatures; i += 1) {
        b[i * outFeatures + o] = weight.data[o * inFeatures + i];
      }
    }
    transposed.set(weight.data, b);
  }
  const out = await backend.matmul(x.data, b, rows, outFeatures, inFeatures);
  if (bias) {
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outFeatures; o += 1) {
        out[r * outFeatures + o] = out[r * outFeatures + o] + bias.data[o];
      }
    }
  }
  return { data: out, shape: [...x.shape.slice(0, -1), outFeatures] };
}
function silu(x) {
  return activation({ input: x, kind: ACTIVATION.silu });
}
function addInPlace(a, b) {
  for (let i = 0; i < a.length; i += 1) a[i] = a[i] + b[i];
  return a;
}
function transpose2d(data, rows, cols) {
  const out = new Float32Array(data.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) out[c * rows + r] = data[r * cols + c];
  }
  return out;
}
function interpolateLinear(input, channels, targetLength) {
  const sourceLength = input.data.length / channels;
  if (sourceLength === targetLength) return input;
  const out = new Float32Array(channels * targetLength);
  const scale = sourceLength / targetLength;
  for (let t = 0; t < targetLength; t += 1) {
    const position = Math.max(0, (t + 0.5) * scale - 0.5);
    const left = Math.min(Math.floor(position), sourceLength - 1);
    const right = Math.min(left + 1, sourceLength - 1);
    const weight = position - left;
    for (let c = 0; c < channels; c += 1) {
      const a = input.data[c * sourceLength + left];
      const b = input.data[c * sourceLength + right];
      out[c * targetLength + t] = a + (b - a) * weight;
    }
  }
  return { data: out, shape: [channels, targetLength] };
}
function windowMask(length, windowSize) {
  const perSide = Math.floor(windowSize / 2);
  const mask = new Float32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      mask[i * length + j] = Math.abs(i - j) <= perSide ? 0 : -Infinity;
    }
  }
  return mask;
}
var Weights = class {
  constructor(file) {
    this.file = file;
  }
  file;
  // `Safetensors.tensor` copies out of the checkpoint on every call, by design
  // — a view would pin the whole file. That makes it the wrong thing to call
  // per layer per forward, which is what an uncached `get` does: the decoder
  // asks for the same two hundred tensors on every decode. Memoised by name, so
  // each is copied once and every later reader gets the same array — which is
  // also what makes the transpose cache below able to key on identity.
  cache = /* @__PURE__ */ new Map();
  get(name) {
    let tensor = this.cache.get(name);
    if (!tensor) {
      const view = this.file.tensor(name);
      tensor = { data: view.data, shape: [...view.shape] };
      this.cache.set(name, tensor);
    }
    return tensor;
  }
  maybe(name) {
    return this.file.has(name) ? this.get(name) : null;
  }
};
async function fsqDecode(tokens, levels, weights, backend) {
  const basis = [];
  let running = 1;
  for (let i = 0; i < levels.length; i += 1) {
    basis.push(running);
    running *= levels[i];
  }
  const codes = new Float32Array(tokens.length * levels.length);
  for (let t = 0; t < tokens.length; t += 1) {
    for (let d = 0; d < levels.length; d += 1) {
      const halfWidth = Math.floor(levels[d] / 2);
      const code = Math.floor(tokens[t] / basis[d]) % levels[d];
      codes[t * levels.length + d] = (code - halfWidth) / halfWidth;
    }
  }
  return await linear(
    { data: codes, shape: [tokens.length, levels.length] },
    weights.get("local_quantizer.proj_out.weight"),
    weights.maybe("local_quantizer.proj_out.bias"),
    backend
  );
}
function layerNorm(x, weight, bias, dim) {
  return {
    data: layernorm({
      input: x.data,
      weight: weight.data,
      bias: bias.data,
      N: x.data.length / dim,
      D: dim,
      eps: NORM_EPS
    }),
    shape: [...x.shape]
  };
}
async function adaLnZero(x, condition, dim, prefix, weights, withGate, backend) {
  const rows = x.data.length / dim;
  const normed = layernorm({
    input: x.data,
    // elementwise_affine=False upstream, so the identity affine here rather
    // than weights that do not exist in the checkpoint.
    weight: new Float32Array(dim).fill(1),
    bias: new Float32Array(dim),
    N: rows,
    D: dim,
    eps: NORM_EPS
  });
  const projected = await linear(
    { data: silu(condition), shape: [1, condition.length] },
    weights.get(`${prefix}.condition_proj.1.weight`),
    weights.maybe(`${prefix}.condition_proj.1.bias`),
    backend
  );
  const parts = withGate ? 3 : 2;
  const shift = projected.data.subarray(0, dim);
  const scale = projected.data.subarray(dim, 2 * dim);
  const gate = withGate ? projected.data.slice(2 * dim, 3 * dim) : null;
  if (projected.data.length !== parts * dim) {
    throw new Error(`${prefix} projected ${projected.data.length}, expected ${parts * dim}`);
  }
  const out = new Float32Array(normed.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < dim; c += 1) {
      const i = r * dim + c;
      out[i] = normed[i] * (1 + scale[c]) + shift[c];
    }
  }
  return { modulated: { data: out, shape: [...x.shape] }, gate };
}
async function selfAttention(x, config, prefix, weights, mask, length, backend) {
  const { dim, heads, ropeTheta } = config;
  const headDim = dim / heads;
  const q = await linear(x, weights.get(`${prefix}.wq.weight`), weights.maybe(`${prefix}.wq.bias`), backend);
  const k = await linear(x, weights.get(`${prefix}.wk.weight`), weights.maybe(`${prefix}.wk.bias`), backend);
  const v = await linear(x, weights.get(`${prefix}.wv.weight`), weights.maybe(`${prefix}.wv.bias`), backend);
  const roped = (t) => ({
    data: rope({
      input: t.data,
      N: length,
      numHeads: heads,
      headDim,
      posOffset: 0,
      thetaBase: ropeTheta
    }),
    shape: t.shape
  });
  const toHeadMajor = (t) => {
    const out = new Float32Array(t.data.length);
    for (let l = 0; l < length; l += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let d = 0; d < headDim; d += 1) {
          out[(h * length + l) * headDim + d] = t.data[(l * heads + h) * headDim + d];
        }
      }
    }
    return out;
  };
  const { output: scores } = attention({
    q: toHeadMajor(roped(q)),
    k: toHeadMajor(roped(k)),
    v: toHeadMajor(v),
    B: 1,
    H: heads,
    L: length,
    S: length,
    D: headDim,
    Dv: headDim,
    // `Attention.scale` is `head_dim ** -0.5`, which is also this op's default;
    // passed anyway so the two cannot drift apart silently.
    scale: 1 / Math.sqrt(headDim),
    mask,
    maskShape: [1, 1, length]
  });
  const merged = new Float32Array(scores.length);
  for (let l = 0; l < length; l += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < headDim; d += 1) {
        merged[(l * heads + h) * headDim + d] = scores[(h * length + l) * headDim + d];
      }
    }
  }
  return await linear(
    { data: merged, shape: [length, dim] },
    weights.get(`${prefix}.wo.weight`),
    weights.maybe(`${prefix}.wo.bias`),
    backend
  );
}
async function feedForward(x, prefix, weights, backend) {
  const gate = await linear(x, weights.get(`${prefix}.w1.weight`), null, backend);
  const up = await linear(x, weights.get(`${prefix}.w3.weight`), null, backend);
  const activated = silu(gate.data);
  for (let i = 0; i < activated.length; i += 1) activated[i] = activated[i] * up.data[i];
  return await linear(
    { data: activated, shape: gate.shape },
    weights.get(`${prefix}.w2.weight`),
    null,
    backend
  );
}
async function transformer(input, config, prefix, weights, condition, backend) {
  const { dim, layers, windowSize } = config;
  const length = input.data.length / dim;
  const mask = windowMask(length, windowSize);
  const useAdaLn = config.adaLnConditionDim !== void 0;
  if (useAdaLn && !condition) throw new Error(`${prefix} needs a condition`);
  let x = { data: Float32Array.from(input.data), shape: [length, dim] };
  for (let layer = 0; layer < layers; layer += 1) {
    const layerPrefix = `${prefix}.layers.${layer}`;
    let normed;
    let attnGate = null;
    if (useAdaLn) {
      const result = await adaLnZero(
        x,
        condition,
        dim,
        `${layerPrefix}.attention_norm`,
        weights,
        true,
        backend
      );
      normed = result.modulated;
      attnGate = result.gate;
    } else {
      normed = layerNorm(
        x,
        weights.get(`${layerPrefix}.attention_norm.weight`),
        weights.get(`${layerPrefix}.attention_norm.bias`),
        dim
      );
    }
    const attended = await selfAttention(
      normed,
      config,
      `${layerPrefix}.attention`,
      weights,
      mask,
      length,
      backend
    );
    applyGated(x.data, attended.data, attnGate, dim);
    let ffnNormed;
    let ffnGate = null;
    if (useAdaLn) {
      const result = await adaLnZero(
        x,
        condition,
        dim,
        `${layerPrefix}.ffn_norm`,
        weights,
        true,
        backend
      );
      ffnNormed = result.modulated;
      ffnGate = result.gate;
    } else {
      ffnNormed = layerNorm(
        x,
        weights.get(`${layerPrefix}.ffn_norm.weight`),
        weights.get(`${layerPrefix}.ffn_norm.bias`),
        dim
      );
    }
    const forwarded = await feedForward(ffnNormed, `${layerPrefix}.feed_forward`, weights, backend);
    applyGated(x.data, forwarded.data, ffnGate, dim);
  }
  let out;
  if (useAdaLn) {
    out = (await adaLnZero(x, condition, dim, `${prefix}.norm`, weights, false, backend)).modulated;
  } else {
    out = layerNorm(x, weights.get(`${prefix}.norm.weight`), weights.get(`${prefix}.norm.bias`), dim);
  }
  const projWeight = weights.maybe(`${prefix}.output_proj.weight`);
  if (projWeight) {
    out = await linear(out, projWeight, weights.maybe(`${prefix}.output_proj.bias`), backend);
  }
  return out;
}
function applyGated(x, y, gate, dim) {
  if (!gate) {
    addInPlace(x, y);
    return;
  }
  const rows = x.length / dim;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < dim; c += 1) {
      x[r * dim + c] = x[r * dim + c] + gate[c] * y[r * dim + c];
    }
  }
}
async function resnetStack(input, channels, length, config, prefix, weights, backend) {
  const kernel = config.waveResnetKernelSize;
  const padding2 = kernel - 1 >> 1;
  let x = Float32Array.from(input.data);
  for (let block = 0; block < config.waveResnetNumBlocks; block += 1) {
    const blockPrefix = `${prefix}.blocks.${block}`;
    const residual = Float32Array.from(x);
    for (const [normName, convName] of [
      ["norm1", "conv1"],
      ["norm2", "conv2"]
    ]) {
      const normed = groupNorm({
        input: x,
        weight: weights.get(`${blockPrefix}.${normName}.weight`).data,
        bias: weights.get(`${blockPrefix}.${normName}.bias`).data,
        N: 1,
        C: channels,
        L: length,
        G: config.waveResnetNumGroups,
        eps: GROUP_NORM_EPS
      });
      const weight = weights.get(`${blockPrefix}.${convName}.weight`);
      x = await backend.conv1d(
        silu(normed),
        weight.data,
        weights.maybe(`${blockPrefix}.${convName}.bias`)?.data ?? null,
        channels,
        channels,
        length,
        kernel,
        padding2
      );
    }
    addInPlace(x, residual);
  }
  return { data: x, shape: [channels, length] };
}
async function decode(tokens, globalEmbedding, stftLength, config, weights, backend = cpuBackend) {
  const stages = {};
  const contentEmbedding = await fsqDecode(tokens, config.fsqLevels, weights, backend);
  stages.content_embedding = contentEmbedding;
  const prenetOut = await transformer(
    contentEmbedding,
    config.prenet,
    "wave_prenet",
    weights,
    null,
    backend
  );
  stages.after_prenet = prenetOut;
  const prenetDim = config.prenet.outputDim ?? config.prenet.dim;
  const upsampleWeight = weights.get("wave_conv_upsample.weight");
  const upsampled = convTranspose1d({
    input: transpose2d(prenetOut.data, tokens.length, prenetDim),
    weight: upsampleWeight.data,
    bias: weights.maybe("wave_conv_upsample.bias")?.data,
    N: 1,
    Cin: prenetDim,
    Cout: prenetDim,
    L: tokens.length,
    K: upsampleWeight.shape[2],
    stride: 2
  });
  const upsampledLength = upsampled.length / prenetDim;
  stages.after_conv_upsample = { data: upsampled, shape: [prenetDim, upsampledLength] };
  const interpolated = interpolateLinear(stages.after_conv_upsample, prenetDim, stftLength);
  stages.after_interpolate = interpolated;
  const dim = config.decoder.dim;
  stages.after_prior_net = await resnetStack(
    interpolated,
    dim,
    stftLength,
    config,
    "wave_prior_net",
    weights,
    backend
  );
  const decoderInput = {
    data: transpose2d(stages.after_prior_net.data, dim, stftLength),
    shape: [stftLength, dim]
  };
  const decoded = await transformer(
    decoderInput,
    config.decoder,
    "wave_decoder",
    weights,
    globalEmbedding,
    backend
  );
  stages.after_decoder = decoded;
  stages.after_post_net = await resnetStack(
    { data: transpose2d(decoded.data, stftLength, dim), shape: [dim, stftLength] },
    dim,
    stftLength,
    config,
    "wave_post_net",
    weights,
    backend
  );
  const headInput = {
    data: transpose2d(stages.after_post_net.data, dim, stftLength),
    shape: [stftLength, dim]
  };
  const projected = await linear(
    headInput,
    weights.get("istft_head.out.weight"),
    weights.maybe("istft_head.out.bias"),
    backend
  );
  stages.istft_linear = projected;
  const bins = config.nFft / 2 + 1;
  const real = new Float32Array(stftLength * bins);
  const imag = new Float32Array(stftLength * bins);
  for (let t = 0; t < stftLength; t += 1) {
    for (let bin = 0; bin < bins; bin += 1) {
      const magnitude = Math.min(Math.exp(projected.data[t * 2 * bins + bin]), 100);
      const phase = projected.data[t * 2 * bins + bins + bin];
      real[t * bins + bin] = magnitude * Math.cos(phase);
      imag[t * bins + bin] = magnitude * Math.sin(phase);
    }
  }
  stages.spec_real = { data: real, shape: [stftLength, bins] };
  stages.spec_imag = { data: imag, shape: [stftLength, bins] };
  const waveform = await backend.istft(
    real,
    imag,
    hannWindow(config.nFft),
    stftLength,
    config.nFft,
    config.hopLength
  );
  stages.waveform = { data: waveform, shape: [waveform.length] };
  return { waveform, stages };
}

// kernels.ts
var MATMUL = "// Matmul (GEMM): C = A @ B, shared-memory tiled.\n//\n// Layout:\n//   a:      [M, K] f32, row-major\n//   b:      [K, N] f32, row-major\n//   output: [M, N] f32, row-major\n//\n// One workgroup owns one TILE x TILE block of C. It walks K a tile at a time,\n// staging A's block and B's block in workgroup memory, so each loaded value is\n// used TILE times instead of once. That reuse is the whole reason this op is\n// separate from GEMV, which has none to find.\n//\n// TILE = 16 is not a measured optimum \u2014 nothing here is tuned yet (see #3/#4\n// for the roofline harness). It is the plain choice that fits the limits with\n// room to grow:\n//   * 16 x 16 = 256 invocations per workgroup, the same width the other ops in\n//     this repo use, and well under maxComputeInvocationsPerWorkgroup (1024).\n//   * two f32 tiles = 2 * 16 * 16 * 4 = 2048 bytes of workgroup storage, against\n//     maxComputeWorkgroupStorageSize (49152), so a later register-blocked or\n//     double-buffered variant has somewhere to go.\n//   * one output per invocation, which keeps the indexing readable. Correctness\n//     first (rule 8); the register blocking that makes this fast comes after\n//     there is a number to improve on.\n// Changing TILE here means changing it in wgsl.test.ts too \u2014 the ragged shapes\n// are chosen around it.\n\nstruct Params {\n  M: u32,\n  N: u32,\n  K: u32,\n}\n\n@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nconst TILE: u32 = 16u;\n\nvar<workgroup> tile_a: array<array<f32, TILE>, TILE>;\nvar<workgroup> tile_b: array<array<f32, TILE>, TILE>;\n\n@compute @workgroup_size(TILE, TILE)\nfn main(\n  @builtin(workgroup_id) wg_id: vec3<u32>,\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n  let lx = local_id.x;\n  let ly = local_id.y;\n  let row = wg_id.y * TILE + ly;\n  let col = wg_id.x * TILE + lx;\n\n  // No early return for invocations off the edge of C. They have no output to\n  // write, but they still have to load their share of the tiles and reach every\n  // barrier \u2014 leaving early would hang the ones that stayed and would leave the\n  // tile half filled.\n  var acc: f32 = 0.0;\n  let k_tiles = (params.K + TILE - 1u) / TILE;\n  for (var t: u32 = 0u; t < k_tiles; t += 1u) {\n    let k_base = t * TILE;\n\n    // The ragged K tail lives here and nowhere else: `k_len` is how much of this\n    // tile is real, and lanes past it are neither written nor read. The obvious\n    // alternative \u2014 pad the tiles with zeros and always run the full TILE \u2014 is\n    // not used, because then the padding and the loop bound each mask the other:\n    // zeroing either factor makes the product vanish, so removing one of them\n    // leaves the tests green and the guard untested. One mechanism, one thing to\n    // break. (This is uniform across the workgroup, so the barriers below stay\n    // in uniform control flow.)\n    let k_len = min(TILE, params.K - k_base);\n\n    // A lane whose row is past M, or whose column is past N, leaves its slot\n    // holding whatever the previous tile left there. That is safe and deliberate:\n    // tile_a[ly][*] is only ever read by lanes with this same `ly` \u2014 the same\n    // row \u2014 and tile_b[*][lx] only by lanes with this same `lx`, so a stale slot\n    // can only reach an accumulator that is thrown away at the store below.\n    if (row < params.M && lx < k_len) {\n      tile_a[ly][lx] = a[row * params.K + k_base + lx];\n    }\n    if (col < params.N && ly < k_len) {\n      tile_b[ly][lx] = b[(k_base + ly) * params.N + col];\n    }\n    workgroupBarrier();\n\n    for (var k: u32 = 0u; k < k_len; k += 1u) {\n      acc += tile_a[ly][k] * tile_b[k][lx];\n    }\n    // Before overwriting the tiles on the next pass, everyone must be done\n    // reading them.\n    workgroupBarrier();\n  }\n\n  // The two halves are not equally load-bearing, and the difference is worth\n  // knowing rather than assuming. `col < N` is the one that matters: past the\n  // last column, row * N + col is C[row + 1][col - N] \u2014 a live element of the\n  // next row, silently overwritten with this invocation's accumulator. Dropping\n  // it turns the N-tail tests red immediately.\n  //\n  // `row < M` is hygiene. Past the last row the index is off the end of the\n  // buffer, and this implementation discards the write, so dropping it leaves\n  // every test green (checked, by mutation). It stays because WGSL does not\n  // promise that an out-of-bounds write is a no-op \u2014 only that it will not\n  // reach another resource.\n  if (row < params.M && col < params.N) {\n    output[row * params.N + col] = acc;\n  }\n}\n";
var CONV1D = "// conv1d, matching torch.nn.functional.conv1d.\n//\n// A cross-correlation, not a true convolution: tap k reads forward from the\n// window start and the kernel is never flipped. See reference.ts for the\n// measurement that settles it.\n//\n// One thread per output element. That is all the parallelism this shape has\n// before tiling, and rule 8 says the plain version has to agree with the\n// reference before anything clever gets written.\n//\n// Layout:\n//   input:  [N, Cin, L]              f32\n//   weight: [Cout, Cin/groups, K]    f32\n//   bias:   [Cout]                   f32 \u2014 required here; PyTorch's bias=None\n//                                    is passed as zeros, which costs one add\n//                                    and saves a branch nothing can observe\n//   output: [N, Cout, Lout]          f32\n//\n// Dispatch: x over Lout in 256-wide workgroups, y = Cout, z = N.\n\nstruct Params {\n  Cin: u32,\n  Cout: u32,\n  L: u32,\n  K: u32,\n  Lout: u32,\n  stride: u32,\n  padding: u32,\n  dilation: u32,\n  // Cin / groups and Cout / groups. The kernel never needs `groups` itself,\n  // only the two sizes it divides into, and dividing on the host keeps an\n  // integer division out of every thread.\n  in_per_group: u32,\n  out_per_group: u32,\n  // A uniform struct rounds up to a multiple of 16 bytes. Named rather than\n  // implied, so the host packing twelve words is obviously deliberate.\n  reserved_0: u32,\n  reserved_1: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read> bias: array<f32>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let ol = gid.x;\n  let oc = gid.y;\n  let n = gid.z;\n\n  // Lout is rarely a multiple of 256, so the last workgroup of each row runs\n  // surplus threads. Unguarded they walk into the next channel's output.\n  if (ol >= params.Lout) {\n    return;\n  }\n\n  let group = oc / params.out_per_group;\n  let ic_base = group * params.in_per_group;\n  // Where this output's window starts in the input, before the pad is trimmed.\n  // Signed: with padding it is negative for the first few outputs.\n  let window = i32(ol * params.stride) - i32(params.padding);\n\n  var acc = bias[oc];\n  for (var ic_local = 0u; ic_local < params.in_per_group; ic_local += 1u) {\n    let in_row = (n * params.Cin + ic_base + ic_local) * params.L;\n    let w_row = (oc * params.in_per_group + ic_local) * params.K;\n    for (var k = 0u; k < params.K; k += 1u) {\n      let il = window + i32(k * params.dilation);\n      // The zero pad, in two halves. Neither can be left to the hardware: this\n      // device reads past the end of a buffer as zero, which is the right\n      // answer by accident, but one row's out-of-range index is the next row's\n      // valid data, and that is what actually comes back.\n      if (il < 0) {\n        continue;\n      }\n      if (il >= i32(params.L)) {\n        continue;\n      }\n      acc += input[in_row + u32(il)] * weight[w_row + k];\n    }\n  }\n\n  output[(n * params.Cout + oc) * params.Lout + ol] = acc;\n}\n";
var ISTFT = "// ISTFT: inverse one-sided DFT per frame, windowed, overlap-added, and divided\n// by the overlap-added w\xB2 envelope. The thing ONNX cannot express.\n//\n// Layout:\n//   real, imag: [frames, bins] f32, frame-major\n//   window:     [nFft] f32\n//   out:        [outLength] f32\n//\n// One thread per **output sample**, gathering rather than scattering. Written\n// this way for a reason: the natural overlap-add scatters frames into a shared\n// buffer, where two frames land on the same sample and the sum needs atomics or\n// a second pass. Reading instead of writing, each output sample is computed by\n// exactly one thread and there is nothing to order. It costs re-deriving the\n// inverse transform once per overlapping frame \u2014 two of them, at 2x overlap.\n//\n// The envelope division is not optional and not an ordinary normalisation: see\n// reference.ts. A periodic Hann at 50% overlap is COLA in w but not in w\xB2, so\n// skipping it is wrong by up to 2x in a way that still sounds like audio.\n\nstruct Params {\n  nFft: u32,\n  hop: u32,\n  bins: u32,\n  frames: u32,\n  outLength: u32,\n  /// floor(nFft/2) when centred, 0 when not. The caller resolves the convention.\n  pad: u32,\n}\n\n@group(0) @binding(0) var<storage, read> real: array<f32>;\n@group(0) @binding(1) var<storage, read> imag: array<f32>;\n@group(0) @binding(2) var<storage, read> win: array<f32>;\n@group(0) @binding(3) var<storage, read_write> out: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst TWO_PI: f32 = 6.28318530717958647692;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let t = global_id.x;\n  if (t >= params.outLength) {\n    return;\n  }\n  let position = i32(t + params.pad);\n  // Even nFft has a Nyquist bin, which stands for itself rather than for a\n  // conjugate pair and so is counted once. Odd nFft has none, and every bin\n  // above DC doubles. torch.fft.irfft splits the same way.\n  let hasNyquist = (params.nFft % 2u) == 0u;\n\n  var numerator: f32 = 0.0;\n  var envelope: f32 = 0.0;\n  for (var frame = 0u; frame < params.frames; frame += 1u) {\n    let n = position - i32(frame * params.hop);\n    // The `n < 0` half of this cannot be caught by a test on this device, and\n    // is written down rather than left to be discovered. Dropping it makes\n    // `u32(n)` wrap to about 4e9; this GPU reads that far past a buffer as\n    // zero, the window value comes back 0, and the frame contributes nothing \u2014\n    // which is accidentally the right answer. WGSL allows an implementation to\n    // clamp the index instead, and a device that clamps would read a real\n    // window value and add a whole frame that does not belong to this sample.\n    if (n < 0 || n >= i32(params.nFft)) {\n      continue;\n    }\n    let base = frame * params.bins;\n    // DC counts once, and its imaginary part is dropped along with Nyquist's.\n    var acc: f32 = real[base];\n    for (var k = 1u; k < params.bins; k += 1u) {\n      let weight = select(2.0, 1.0, hasNyquist && k == params.bins - 1u);\n      // Folded into one turn as an integer, as in the forward kernel.\n      let angle = TWO_PI * (f32((k * u32(n)) % params.nFft) / f32(params.nFft));\n      acc += weight * (real[base + k] * cos(angle) - imag[base + k] * sin(angle));\n    }\n    let w = win[u32(n)];\n    numerator += w * (acc / f32(params.nFft));\n    envelope += w * w;\n  }\n  // No NOLA guard here. A shader cannot raise, and a guard that silently\n  // substituted a value would hand back a waveform for a window that cannot\n  // reconstruct one. The reference refuses those inputs before they get here.\n  out[t] = numerator / envelope;\n}\n";

// gpu.ts
var TILE = 16;
var WORKGROUP = 256;
var Gpu = class _Gpu {
  constructor(device, adapterInfo) {
    this.device = device;
    this.adapterInfo = adapterInfo;
  }
  device;
  adapterInfo;
  pipelines = /* @__PURE__ */ new Map();
  /**
   * Keyed on the weight's own array, so a caller that hands over the same
   * tensor twice uploads once. `WeakMap`, so dropping the checkpoint drops the
   * buffers with it rather than pinning half a gigabyte of VRAM behind a cache
   * nobody can reach.
   */
  resident = /* @__PURE__ */ new WeakMap();
  /**
   * A device, or null where WebGPU is absent or refuses.
   *
   * Null rather than a throw: the demo has a working CPU path and falling back
   * to it is a better answer than a broken page. What must not happen is
   * falling back **silently** — the caller reports which one ran.
   */
  /**
   * Wrap a device obtained some other way.
   *
   * Node has no `navigator.gpu`; the `webgpu` package hands back a `GPU` after
   * installing its globals, and web-xpu-ops' own harness uses it that way. The
   * tests need a real device rather than a mock — a kernel that compiles and
   * computes the wrong thing is exactly what a mock cannot catch.
   */
  static fromDevice(device, info) {
    return new _Gpu(device, info);
  }
  static async create() {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
      }
    });
    const info = adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" ") || "unknown adapter" : "unknown adapter";
    return new _Gpu(device, info);
  }
  pipeline(code) {
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" }
      });
      this.pipelines.set(code, pipeline);
    }
    return pipeline;
  }
  upload(data) {
    const buffer = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  }
  /** Upload once and keep, for anything that does not change between calls. */
  residentBuffer(data) {
    let buffer = this.resident.get(data);
    if (!buffer) {
      buffer = this.upload(data);
      this.resident.set(data, buffer);
    }
    return buffer;
  }
  uniform(values) {
    const words = new Uint32Array(Math.max(4, Math.ceil(values.length / 4) * 4));
    words.set(values);
    const buffer = this.device.createBuffer({
      size: words.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(buffer, 0, words);
    return buffer;
  }
  async dispatch(code, inputs, outputLength, uniforms, workgroups) {
    const device = this.device;
    const pipeline = this.pipeline(code);
    const output = device.createBuffer({
      size: Math.max(4, outputLength * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const params = this.uniform(uniforms);
    const entries = [];
    inputs.forEach((buffer, index) => entries.push({ binding: index, resource: { buffer } }));
    entries.push({ binding: inputs.length, resource: { buffer: output } });
    entries.push({ binding: inputs.length + 1, resource: { buffer: params } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const staging = device.createBuffer({
      size: Math.max(4, outputLength * 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, Math.max(4, outputLength * 4));
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0, outputLength * 4));
    staging.unmap();
    staging.destroy();
    output.destroy();
    params.destroy();
    return result;
  }
  /**
   * `[M, K] x [K, N]`, with `b` kept on the device between calls.
   *
   * `b` is the transposed weight, which is the same array every time this layer
   * runs; `a` is the activation, which is not.
   */
  async matmul(a, b, M, N, K) {
    const bBuffer = this.residentBuffer(b);
    const aBuffer = this.upload(a);
    try {
      return await this.dispatch(MATMUL, [aBuffer, bBuffer], M * N, [M, N, K], [
        Math.ceil(N / TILE),
        Math.ceil(M / TILE)
      ]);
    } finally {
      aBuffer.destroy();
    }
  }
  /** `conv1d`, with the weight and bias resident. `N` is always 1 here. */
  async conv1d(input, weight, bias, Cin, Cout, L, K, padding2) {
    const outLength = L + 2 * padding2 - (K - 1) - 1 + 1;
    const inputBuffer = this.upload(input);
    const weightBuffer = this.residentBuffer(weight);
    const biasBuffer = this.residentBuffer(bias ?? zeros(Cout));
    try {
      const dispatchLength = Math.ceil(outLength / WORKGROUP) * WORKGROUP;
      return await this.dispatch(
        CONV1D,
        [inputBuffer, weightBuffer, biasBuffer],
        Cout * outLength,
        [Cin, Cout, L, K, outLength, 1, padding2, 1, Cin, Cout, 0, 0],
        [dispatchLength / WORKGROUP, Cout, 1]
      );
    } finally {
      inputBuffer.destroy();
    }
  }
  /** The inverse transform. `pad` carries the padding convention, resolved by the caller. */
  async istft(real, imag, window, frames, nFft, hop, pad, outLength) {
    const bins = Math.floor(nFft / 2) + 1;
    const realBuffer = this.upload(real);
    const imagBuffer = this.upload(imag);
    const windowBuffer = this.residentBuffer(window);
    try {
      return await this.dispatch(
        ISTFT,
        [realBuffer, imagBuffer, windowBuffer],
        outLength,
        [nFft, hop, bins, frames, outLength, pad],
        [Math.ceil(outLength / WORKGROUP)]
      );
    } finally {
      realBuffer.destroy();
      imagBuffer.destroy();
    }
  }
  destroy() {
    this.device.destroy();
  }
};
var ZEROS = /* @__PURE__ */ new Map();
function zeros(length) {
  let array = ZEROS.get(length);
  if (!array) {
    array = new Float32Array(length);
    ZEROS.set(length, array);
  }
  return array;
}
function gpuBackend(gpu) {
  return {
    name: `WebGPU (${gpu.adapterInfo})`,
    matmul: (a, b, M, N, K) => gpu.matmul(a, b, M, N, K),
    conv1d: (input, weight, bias, Cin, Cout, L, K, padding2) => gpu.conv1d(input, weight, bias, Cin, Cout, L, K, padding2),
    istft: (real, imag, window, frames, nFft, hop) => (
      // `"same"` resolved here, because the kernel takes a number and has no
      // convention of its own: crop `(nFft - hop) / 2` from each end, which
      // leaves `hop * frames` samples. The reference does the same arithmetic
      // behind the mode name.
      gpu.istft(real, imag, window, frames, nFft, hop, (nFft - hop) / 2, hop * frames)
    )
  };
}

// safetensors.ts
var Safetensors = class _Safetensors {
  constructor(header, buffer, dataStart) {
    this.header = header;
    this.buffer = buffer;
    this.dataStart = dataStart;
  }
  header;
  buffer;
  dataStart;
  static parse(buffer) {
    if (buffer.byteLength < 8) {
      throw new Error(`not safetensors: ${buffer.byteLength} bytes is shorter than the header length`);
    }
    const view = new DataView(buffer);
    const headerLength = Number(view.getBigUint64(0, true));
    if (headerLength <= 0 || 8 + headerLength > buffer.byteLength) {
      throw new Error(`header claims ${headerLength} bytes, file has ${buffer.byteLength}`);
    }
    const json = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength));
    const header = JSON.parse(json);
    delete header.__metadata__;
    return new _Safetensors(header, buffer, 8 + headerLength);
  }
  names() {
    return Object.keys(this.header);
  }
  has(name) {
    return name in this.header;
  }
  /**
   * One tensor as f32.
   *
   * Only `F32` is accepted. The alternative — silently widening an `F16` or a
   * `BF16` — would produce a tensor of entirely plausible numbers carrying half
   * the precision the caller assumed, and every later comparison would be
   * chasing that instead of the bug it was written for. This checkpoint is f32
   * throughout; a rung that is not should say so here rather than downstream.
   */
  tensor(name) {
    const entry = this.header[name];
    if (!entry) {
      throw new Error(`no tensor "${name}" in the checkpoint`);
    }
    if (entry.dtype !== "F32") {
      throw new Error(`"${name}" is ${entry.dtype}; only F32 is read here`);
    }
    const [start, end] = entry.data_offsets;
    const count = entry.shape.reduce((a, b) => a * b, 1);
    if (end - start !== count * 4) {
      throw new Error(
        `"${name}" spans ${end - start} bytes but its shape ${entry.shape.join("x")} needs ${count * 4}`
      );
    }
    const bytes = new Uint8Array(this.buffer, this.dataStart + start, end - start);
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return { data: new Float32Array(copy.buffer), shape: entry.shape };
  }
};

// browser.ts
var CHECKPOINT = "https://huggingface.co/Aratako/MioCodec-25Hz-24kHz/resolve/main/model.safetensors";
async function fetchCheckpoint(report) {
  report("downloading the checkpoint");
  const response = await fetch(CHECKPOINT);
  if (!response.ok) throw new Error(`checkpoint: HTTP ${response.status}`);
  const header = response.headers.get("content-length");
  const total = header ? Number(header) : void 0;
  const reader = response.body?.getReader();
  if (!reader) return await response.arrayBuffer();
  const chunks = [];
  let loaded = 0;
  for (; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    report("downloading the checkpoint", { loaded, total });
  }
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}
async function chooseBackend(choice) {
  if (choice === "cpu") return { backend: cpuBackend, gpu: null };
  const gpu = await Gpu.create();
  if (gpu) return { backend: gpuBackend(gpu), gpu };
  if (choice === "gpu") throw new Error("WebGPU is unavailable, and the GPU backend was required");
  return { backend: cpuBackend, gpu: null };
}
async function run(fixture, report, choice = "auto") {
  const buffer = await fetchCheckpoint(report);
  report("parsing the checkpoint");
  const weights = new Weights(Safetensors.parse(buffer));
  const { backend, gpu } = await chooseBackend(choice);
  report(`decoding on ${backend.name}`);
  await new Promise((resolve2) => setTimeout(resolve2, 0));
  try {
    const started = performance.now();
    const { waveform } = await decode(
      Float32Array.from(fixture.tokens),
      Float32Array.from(fixture.global_embedding),
      fixture.stft_length,
      MIOCODEC_24K,
      weights,
      backend
    );
    const elapsedMs = performance.now() - started;
    const seconds = waveform.length / fixture.sample_rate;
    return {
      pcm: waveform,
      sampleRate: fixture.sample_rate,
      seconds,
      elapsedMs,
      realTimeFactor: elapsedMs / 1e3 / seconds,
      backend: backend.name
    };
  } finally {
    gpu?.destroy();
  }
}
function toWav(pcm, sampleRate) {
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
export {
  fetchCheckpoint,
  run,
  toWav
};
