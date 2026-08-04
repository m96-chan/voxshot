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
function linear(x, weight, bias) {
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
  const out = matmul({ a: x.data, b, M: rows, N: outFeatures, K: inFeatures });
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
function fsqDecode(tokens, levels, weights) {
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
  return linear(
    { data: codes, shape: [tokens.length, levels.length] },
    weights.get("local_quantizer.proj_out.weight"),
    weights.maybe("local_quantizer.proj_out.bias")
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
function adaLnZero(x, condition, dim, prefix, weights, withGate) {
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
  const projected = linear(
    { data: silu(condition), shape: [1, condition.length] },
    weights.get(`${prefix}.condition_proj.1.weight`),
    weights.maybe(`${prefix}.condition_proj.1.bias`)
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
function selfAttention(x, config, prefix, weights, mask, length) {
  const { dim, heads, ropeTheta } = config;
  const headDim = dim / heads;
  const q = linear(x, weights.get(`${prefix}.wq.weight`), weights.maybe(`${prefix}.wq.bias`));
  const k = linear(x, weights.get(`${prefix}.wk.weight`), weights.maybe(`${prefix}.wk.bias`));
  const v = linear(x, weights.get(`${prefix}.wv.weight`), weights.maybe(`${prefix}.wv.bias`));
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
  return linear(
    { data: merged, shape: [length, dim] },
    weights.get(`${prefix}.wo.weight`),
    weights.maybe(`${prefix}.wo.bias`)
  );
}
function feedForward(x, prefix, weights) {
  const gate = linear(x, weights.get(`${prefix}.w1.weight`), null);
  const up = linear(x, weights.get(`${prefix}.w3.weight`), null);
  const activated = silu(gate.data);
  for (let i = 0; i < activated.length; i += 1) activated[i] = activated[i] * up.data[i];
  return linear({ data: activated, shape: gate.shape }, weights.get(`${prefix}.w2.weight`), null);
}
function transformer(input, config, prefix, weights, condition) {
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
      const result = adaLnZero(x, condition, dim, `${layerPrefix}.attention_norm`, weights, true);
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
    const attended = selfAttention(normed, config, `${layerPrefix}.attention`, weights, mask, length);
    applyGated(x.data, attended.data, attnGate, dim);
    let ffnNormed;
    let ffnGate = null;
    if (useAdaLn) {
      const result = adaLnZero(x, condition, dim, `${layerPrefix}.ffn_norm`, weights, true);
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
    const forwarded = feedForward(ffnNormed, `${layerPrefix}.feed_forward`, weights);
    applyGated(x.data, forwarded.data, ffnGate, dim);
  }
  let out;
  if (useAdaLn) {
    out = adaLnZero(x, condition, dim, `${prefix}.norm`, weights, false).modulated;
  } else {
    out = layerNorm(x, weights.get(`${prefix}.norm.weight`), weights.get(`${prefix}.norm.bias`), dim);
  }
  const projWeight = weights.maybe(`${prefix}.output_proj.weight`);
  if (projWeight) {
    out = linear(out, projWeight, weights.maybe(`${prefix}.output_proj.bias`));
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
function resnetStack(input, channels, length, config, prefix, weights) {
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
      x = conv1d({
        input: silu(normed),
        weight: weight.data,
        bias: weights.maybe(`${blockPrefix}.${convName}.bias`)?.data,
        N: 1,
        Cin: channels,
        Cout: channels,
        L: length,
        K: kernel,
        padding: padding2
      });
    }
    addInPlace(x, residual);
  }
  return { data: x, shape: [channels, length] };
}
function decode(tokens, globalEmbedding, stftLength, config, weights) {
  const stages = {};
  const contentEmbedding = fsqDecode(tokens, config.fsqLevels, weights);
  stages.content_embedding = contentEmbedding;
  const prenetOut = transformer(contentEmbedding, config.prenet, "wave_prenet", weights, null);
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
  stages.after_prior_net = resnetStack(
    interpolated,
    dim,
    stftLength,
    config,
    "wave_prior_net",
    weights
  );
  const decoderInput = {
    data: transpose2d(stages.after_prior_net.data, dim, stftLength),
    shape: [stftLength, dim]
  };
  const decoded = transformer(decoderInput, config.decoder, "wave_decoder", weights, globalEmbedding);
  stages.after_decoder = decoded;
  stages.after_post_net = resnetStack(
    { data: transpose2d(decoded.data, stftLength, dim), shape: [dim, stftLength] },
    dim,
    stftLength,
    config,
    "wave_post_net",
    weights
  );
  const headInput = {
    data: transpose2d(stages.after_post_net.data, dim, stftLength),
    shape: [stftLength, dim]
  };
  const projected = linear(
    headInput,
    weights.get("istft_head.out.weight"),
    weights.maybe("istft_head.out.bias")
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
  const waveform = istft({
    real,
    imag,
    frames: stftLength,
    nFft: config.nFft,
    hop: config.hopLength,
    window: hannWindow(config.nFft),
    padding: "same"
  });
  stages.waveform = { data: waveform, shape: [waveform.length] };
  return { waveform, stages };
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
async function run(fixture, report) {
  const buffer = await fetchCheckpoint(report);
  report("parsing the checkpoint");
  const weights = new Weights(Safetensors.parse(buffer));
  report("decoding");
  await new Promise((resolve2) => setTimeout(resolve2, 0));
  const started = performance.now();
  const { waveform } = decode(
    Float32Array.from(fixture.tokens),
    Float32Array.from(fixture.global_embedding),
    fixture.stft_length,
    MIOCODEC_24K,
    weights
  );
  const elapsedMs = performance.now() - started;
  const seconds = waveform.length / fixture.sample_rate;
  return {
    pcm: waveform,
    sampleRate: fixture.sample_rate,
    seconds,
    elapsedMs,
    // The number #106 exists to produce. Reported rather than hidden: these are
    // the library's *reference* implementations, whose own README says speed is
    // unmeasured for every op, so this is a floor and not a verdict on WebGPU.
    realTimeFactor: elapsedMs / 1e3 / seconds
  };
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
