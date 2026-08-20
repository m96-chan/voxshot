// ../../../web-xpu-ops/llm/sampler.ts
function applyMask(logits, allowed) {
  const masked = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) masked[i] = logits[i];
  if (allowed === null) return masked;
  for (let i = 0; i < masked.length; i += 1) {
    if (!allowed.has(i)) masked[i] = -Infinity;
  }
  return masked;
}
function argmax(logits) {
  let bestIndex = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = logits[i];
    if (v > bestValue) {
      bestValue = v;
      bestIndex = i;
    }
  }
  return bestIndex;
}
function sampleTopP(logits, temperature, topP, rng) {
  if (!(temperature > 0)) {
    throw new Error(`sampleTopP: temperature must be > 0, got ${temperature}`);
  }
  if (!(topP > 0) || topP > 1) {
    throw new Error(`sampleTopP: topP must be in (0, 1], got ${topP}`);
  }
  const n = logits.length;
  const scaled = new Float64Array(n);
  let maxScaled = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const v = logits[i] / temperature;
    scaled[i] = v;
    if (v > maxScaled) maxScaled = v;
  }
  if (maxScaled === -Infinity) {
    throw new Error("sampleTopP: no finite logit to choose from");
  }
  const probs = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const p = Math.exp(scaled[i] - maxScaled);
    probs[i] = p;
    total += p;
  }
  const order = Array.from({ length: n }, (_, i) => i).filter((i) => probs[i] > 0).sort((a, b) => probs[b] - probs[a]);
  let cumulative = 0;
  let cutoff = order.length;
  for (let k = 0; k < order.length; k += 1) {
    cumulative += probs[order[k]];
    if (cumulative / total >= topP) {
      cutoff = k + 1;
      break;
    }
  }
  const nucleus = order.slice(0, cutoff);
  let nucleusTotal = 0;
  for (const i of nucleus) nucleusTotal += probs[i];
  const draw = rng() * nucleusTotal;
  let acc = 0;
  for (const i of nucleus) {
    acc += probs[i];
    if (draw < acc) return i;
  }
  return nucleus[nucleus.length - 1];
}
function sampleNext(logits, prefixTokens, options, constraint) {
  const allowed = constraint ? constraint.nextAllowed(prefixTokens) : null;
  if (allowed !== null && allowed.size === 0) {
    throw new Error("sampleNext: constraint allows no token to follow this prefix");
  }
  const masked = applyMask(logits, allowed);
  if (options.mode === "greedy") {
    const choice = argmax(masked);
    if (choice < 0) throw new Error("sampleNext: no finite logit to choose from");
    return choice;
  }
  return sampleTopP(masked, options.temperature, options.topP, options.rng ?? Math.random);
}

// ../../../web-xpu-ops/dist/ops/activation/reference.js
var ACTIVATION = {
  relu2: 0,
  silu: 1,
  elu: 2,
  tanh: 3,
  gelu: 4,
  gelu_tanh: 5
};
function erf(x) {
  if (x < 0)
    return -erf(-x);
  if (x >= 6)
    return 1;
  const twoXSquared = 2 * x * x;
  let term = 1;
  let sum = 1;
  for (let n = 1; n < 400; n += 1) {
    term *= twoXSquared / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-18)
      break;
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

// ../../../web-xpu-ops/dist/ops/attention/reference.js
function defaultScale(D) {
  return 1 / Math.sqrt(D);
}
function resolveMask(args, op) {
  const { mask, B, H, L, S } = args;
  const shape = args.maskShape ?? [B, 1, 1];
  if (!mask) {
    if (args.maskShape)
      throw new Error(`${op}(): maskShape given without a mask`);
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
          for (let d = 0; d < D; d += 1)
            dot += q[qHead + i * D + d] * k[kHead + j * D + d];
          row[j] = dot * scale + bias(b, h, i, j);
        }
        let max = -Infinity;
        for (let j = 0; j < S; j += 1)
          max = Math.max(max, row[j]);
        if (max === -Infinity)
          continue;
        let sum = 0;
        for (let j = 0; j < S; j += 1)
          sum += Math.exp(row[j] - max);
        for (let j = 0; j < S; j += 1)
          probs[pHead + i * S + j] = Math.exp(row[j] - max) / sum;
        for (let c = 0; c < Dv; c += 1) {
          let acc = 0;
          for (let j = 0; j < S; j += 1)
            acc += probs[pHead + i * S + j] * v[vHead + j * Dv + c];
          output[oHead + i * Dv + c] = acc;
        }
      }
    }
  }
  return { probs, output };
}

// ../../../web-xpu-ops/dist/ops/conv/reference.js
function conv1dOutputLength({ L, K, stride = 1, padding: padding2 = 0, dilation = 1 }) {
  return Math.floor((L + 2 * padding2 - dilation * (K - 1) - 1) / stride) + 1;
}
function conv1d({ input, weight, bias, N, Cin, Cout, L, K, stride = 1, padding: padding2 = 0, dilation = 1, groups = 1 }) {
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
            if (il < 0 || il >= L)
              continue;
            acc += input[(n * Cin + ic) * L + il] * weight[(oc * inPerGroup + icLocal) * K + k];
          }
        }
        output[(n * Cout + oc) * Lout + ol] = acc;
      }
    }
  }
  return output;
}

// ../../../web-xpu-ops/dist/ops/conv_transpose/reference.js
function convTranspose1dOutputLength({ L, K, stride = 1, padding: padding2 = 0, outputPadding = 0, dilation = 1 }) {
  return (L - 1) * stride - 2 * padding2 + dilation * (K - 1) + outputPadding + 1;
}
function convTranspose1d({ input, weight, bias, N, Cin, Cout, L, K, stride = 1, padding: padding2 = 0, outputPadding = 0, dilation = 1, groups = 1 }) {
  if (padding2 < 0) {
    throw new Error(`convTranspose1d(): negative padding is not supported, got padding=${padding2}`);
  }
  if (outputPadding < 0 || outputPadding >= stride && outputPadding >= dilation) {
    throw new Error(`convTranspose1d(): output_padding=${outputPadding} must be smaller than either stride=${stride} or dilation=${dilation}`);
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
            if (ol < 0 || ol >= Lout)
              continue;
            output[(n * Cout + oc) * Lout + ol] += x * weight[(ic * outPerGroup + ocLocal) * K + k];
          }
        }
      }
    }
  }
  return output;
}

// ../../../web-xpu-ops/dist/ops/group_norm/reference.js
function groupNorm({ input, weight, bias, N, C, L, G, eps }) {
  if (G <= 0 || C % G !== 0) {
    throw new Error(`group_norm: expected number of channels (${C}) to be divisible by num_groups (${G})`);
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

// ../../../web-xpu-ops/dist/ops/layernorm/reference.js
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

// ../../../web-xpu-ops/dist/ops/matmul/reference.js
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

// ../../../web-xpu-ops/dist/ops/rope/reference.js
var UNSCALED = {
  interpolationFactor: 1,
  rampLow: 0,
  rampHigh: 1,
  attentionFactor: 1
};
function ropeFrequencyParams(headDim, thetaBase, scaling) {
  if (!scaling)
    return { ...UNSCALED, effectiveBase: thetaBase };
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
  if (rampHigh === rampLow)
    rampHigh += 1e-3;
  return {
    effectiveBase: thetaBase,
    interpolationFactor: factor,
    rampLow,
    rampHigh,
    attentionFactor: scaling.attentionFactor ?? (factor <= 1 ? 1 : 0.1 * Math.log(factor) + 1)
  };
}
function invFreq({ effectiveBase, interpolationFactor, rampLow, rampHigh }, headDim, pair) {
  const extrapolation = Math.pow(effectiveBase, -2 * pair / headDim);
  const interpolation = extrapolation / interpolationFactor;
  const ramp = Math.min(Math.max((pair - rampLow) / (rampHigh - rampLow), 0), 1);
  return extrapolation + (interpolation - extrapolation) * ramp;
}
function rope({ input, N, numHeads, headDim, posOffset, thetaBase, scaling, cache, headOffset = 0, headCount = numHeads }) {
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
        throw new Error(`rope: cache was built with ${key}=${cache.freq[key]}, called with ${key}=${freq[key]}`);
      }
    }
  }
  if (headOffset < 0 || headCount < 0 || headOffset + headCount > numHeads) {
    throw new Error(`rope: head range [${headOffset}, ${headOffset + headCount}) does not fit ${numHeads} heads`);
  }
  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      if (head < headOffset || head >= headOffset + headCount) {
        const from = (token * numHeads + head) * headDim;
        for (let i = 0; i < headDim; i += 1)
          output[from + i] = input[from + i];
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

// ../../../web-xpu-ops/dist/ops/stft/reference.js
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
  if (mode === true)
    return "center";
  if (mode === false)
    return "none";
  return mode;
}
function padding(nFft, hop, mode) {
  const resolved = resolve(mode);
  if (resolved === "none")
    return 0;
  if (resolved === "same")
    return Math.floor((nFft - hop) / 2);
  return Math.floor(nFft / 2);
}
function hannWindow(n, periodic = true) {
  const denominator = periodic ? n : n - 1;
  return Float32Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / denominator));
}
function istft({ real, imag, frames, nFft, hop, window: window2, center, padding: mode, length }) {
  if (center !== void 0 && mode !== void 0) {
    throw new Error("give either center or padding, not both");
  }
  const resolved = resolve(mode ?? center ?? true);
  if (nFft < 1)
    throw new Error(`nFft must be positive, got ${nFft}`);
  if (hop < 1)
    throw new Error(`hop must be positive, got ${hop}`);
  if (window2.length !== nFft)
    throw new Error(`window must be ${nFft} long, got ${window2.length}`);
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
      if (n < 0 || n >= nFft)
        continue;
      const base = frame * bins;
      let acc = real[base];
      for (let k = 1; k < bins; k += 1) {
        const weight = even && k === bins - 1 ? 1 : 2;
        const angle = 2 * Math.PI * (k * n % nFft) / nFft;
        acc += weight * (real[base + k] * Math.cos(angle) - imag[base + k] * Math.sin(angle));
      }
      const w = window2[n];
      numerator += w * (acc / nFft);
      envelope += w * w;
    }
    if (envelope < NOLA_FLOOR) {
      throw new Error(`window fails NOLA at sample ${t}: the w\xB2 envelope is ${envelope}, below ${NOLA_FLOOR}`);
    }
    output[t] = numerator / envelope;
  }
  return output;
}
var NOLA_FLOOR = 1e-11;

// ../miocodec/decoder.ts
var cpuBackend = {
  name: "reference (CPU)",
  async matmul(a, b, M, N, K) {
    return matmul({ a, b, M, N, K });
  },
  async conv1d(input, weight, bias, Cin, Cout, L, K, padding2) {
    return conv1d({ input, weight, bias: bias ?? void 0, N: 1, Cin, Cout, L, K, padding: padding2 });
  },
  async istft(real, imag, window2, frames, nFft, hop) {
    return istft({ real, imag, frames, nFft, hop, window: window2, padding: "same" });
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

// ../miocodec/kernels.ts
var MATMUL = "// Matmul (GEMM): C = A @ B, shared-memory tiled.\n//\n// Layout:\n//   a:      [M, K] f32, row-major\n//   b:      [K, N] f32, row-major\n//   output: [M, N] f32, row-major\n//\n// One workgroup owns one TILE x TILE block of C. It walks K a tile at a time,\n// staging A's block and B's block in workgroup memory, so each loaded value is\n// used TILE times instead of once. That reuse is the whole reason this op is\n// separate from GEMV, which has none to find.\n//\n// TILE = 16 is not a measured optimum \u2014 nothing here is tuned yet (see #3/#4\n// for the roofline harness). It is the plain choice that fits the limits with\n// room to grow:\n//   * 16 x 16 = 256 invocations per workgroup, the same width the other ops in\n//     this repo use, and well under maxComputeInvocationsPerWorkgroup (1024).\n//   * two f32 tiles = 2 * 16 * 16 * 4 = 2048 bytes of workgroup storage, against\n//     maxComputeWorkgroupStorageSize (49152), so a later register-blocked or\n//     double-buffered variant has somewhere to go.\n//   * one output per invocation, which keeps the indexing readable. Correctness\n//     first (rule 8); the register blocking that makes this fast comes after\n//     there is a number to improve on.\n// Changing TILE here means changing it in wgsl.test.ts too \u2014 the ragged shapes\n// are chosen around it.\n\nstruct Params {\n  M: u32,\n  N: u32,\n  K: u32,\n}\n\n@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nconst TILE: u32 = 16u;\n\nvar<workgroup> tile_a: array<array<f32, TILE>, TILE>;\nvar<workgroup> tile_b: array<array<f32, TILE>, TILE>;\n\n@compute @workgroup_size(TILE, TILE)\nfn main(\n  @builtin(workgroup_id) wg_id: vec3<u32>,\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n  let lx = local_id.x;\n  let ly = local_id.y;\n  let row = wg_id.y * TILE + ly;\n  let col = wg_id.x * TILE + lx;\n\n  // No early return for invocations off the edge of C. They have no output to\n  // write, but they still have to load their share of the tiles and reach every\n  // barrier \u2014 leaving early would hang the ones that stayed and would leave the\n  // tile half filled.\n  var acc: f32 = 0.0;\n  let k_tiles = (params.K + TILE - 1u) / TILE;\n  for (var t: u32 = 0u; t < k_tiles; t += 1u) {\n    let k_base = t * TILE;\n\n    // The ragged K tail lives here and nowhere else: `k_len` is how much of this\n    // tile is real, and lanes past it are neither written nor read. The obvious\n    // alternative \u2014 pad the tiles with zeros and always run the full TILE \u2014 is\n    // not used, because then the padding and the loop bound each mask the other:\n    // zeroing either factor makes the product vanish, so removing one of them\n    // leaves the tests green and the guard untested. One mechanism, one thing to\n    // break. (This is uniform across the workgroup, so the barriers below stay\n    // in uniform control flow.)\n    let k_len = min(TILE, params.K - k_base);\n\n    // A lane whose row is past M, or whose column is past N, leaves its slot\n    // holding whatever the previous tile left there. That is safe and deliberate:\n    // tile_a[ly][*] is only ever read by lanes with this same `ly` \u2014 the same\n    // row \u2014 and tile_b[*][lx] only by lanes with this same `lx`, so a stale slot\n    // can only reach an accumulator that is thrown away at the store below.\n    if (row < params.M && lx < k_len) {\n      tile_a[ly][lx] = a[row * params.K + k_base + lx];\n    }\n    if (col < params.N && ly < k_len) {\n      tile_b[ly][lx] = b[(k_base + ly) * params.N + col];\n    }\n    workgroupBarrier();\n\n    for (var k: u32 = 0u; k < k_len; k += 1u) {\n      acc += tile_a[ly][k] * tile_b[k][lx];\n    }\n    // Before overwriting the tiles on the next pass, everyone must be done\n    // reading them.\n    workgroupBarrier();\n  }\n\n  // The two halves are not equally load-bearing, and the difference is worth\n  // knowing rather than assuming. `col < N` is the one that matters: past the\n  // last column, row * N + col is C[row + 1][col - N] \u2014 a live element of the\n  // next row, silently overwritten with this invocation's accumulator. Dropping\n  // it turns the N-tail tests red immediately.\n  //\n  // `row < M` is hygiene. Past the last row the index is off the end of the\n  // buffer, and this implementation discards the write, so dropping it leaves\n  // every test green (checked, by mutation). It stays because WGSL does not\n  // promise that an out-of-bounds write is a no-op \u2014 only that it will not\n  // reach another resource.\n  if (row < params.M && col < params.N) {\n    output[row * params.N + col] = acc;\n  }\n}\n";
var CONV1D = "// conv1d, matching torch.nn.functional.conv1d.\n//\n// A cross-correlation, not a true convolution: tap k reads forward from the\n// window start and the kernel is never flipped. See reference.ts for the\n// measurement that settles it.\n//\n// One thread per output element. That is all the parallelism this shape has\n// before tiling, and rule 8 says the plain version has to agree with the\n// reference before anything clever gets written.\n//\n// Layout:\n//   input:  [N, Cin, L]              f32\n//   weight: [Cout, Cin/groups, K]    f32\n//   bias:   [Cout]                   f32 \u2014 required here; PyTorch's bias=None\n//                                    is passed as zeros, which costs one add\n//                                    and saves a branch nothing can observe\n//   output: [N, Cout, Lout]          f32\n//\n// Dispatch: x over Lout in 256-wide workgroups, y = Cout, z = N.\n\nstruct Params {\n  Cin: u32,\n  Cout: u32,\n  L: u32,\n  K: u32,\n  Lout: u32,\n  stride: u32,\n  padding: u32,\n  dilation: u32,\n  // Cin / groups and Cout / groups. The kernel never needs `groups` itself,\n  // only the two sizes it divides into, and dividing on the host keeps an\n  // integer division out of every thread.\n  in_per_group: u32,\n  out_per_group: u32,\n  // A uniform struct rounds up to a multiple of 16 bytes. Named rather than\n  // implied, so the host packing twelve words is obviously deliberate.\n  reserved_0: u32,\n  reserved_1: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read> bias: array<f32>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let ol = gid.x;\n  let oc = gid.y;\n  let n = gid.z;\n\n  // Lout is rarely a multiple of 256, so the last workgroup of each row runs\n  // surplus threads. Unguarded they walk into the next channel's output.\n  if (ol >= params.Lout) {\n    return;\n  }\n\n  let group = oc / params.out_per_group;\n  let ic_base = group * params.in_per_group;\n  // Where this output's window starts in the input, before the pad is trimmed.\n  // Signed: with padding it is negative for the first few outputs.\n  let window = i32(ol * params.stride) - i32(params.padding);\n\n  var acc = bias[oc];\n  for (var ic_local = 0u; ic_local < params.in_per_group; ic_local += 1u) {\n    let in_row = (n * params.Cin + ic_base + ic_local) * params.L;\n    let w_row = (oc * params.in_per_group + ic_local) * params.K;\n    for (var k = 0u; k < params.K; k += 1u) {\n      let il = window + i32(k * params.dilation);\n      // The zero pad, in two halves. Neither can be left to the hardware: this\n      // device reads past the end of a buffer as zero, which is the right\n      // answer by accident, but one row's out-of-range index is the next row's\n      // valid data, and that is what actually comes back.\n      if (il < 0) {\n        continue;\n      }\n      if (il >= i32(params.L)) {\n        continue;\n      }\n      acc += input[in_row + u32(il)] * weight[w_row + k];\n    }\n  }\n\n  output[(n * params.Cout + oc) * params.Lout + ol] = acc;\n}\n";
var ISTFT = "// ISTFT: inverse one-sided DFT per frame, windowed, overlap-added, and divided\n// by the overlap-added w\xB2 envelope. The thing ONNX cannot express.\n//\n// Layout:\n//   real, imag: [frames, bins] f32, frame-major\n//   window:     [nFft] f32\n//   out:        [outLength] f32\n//\n// One thread per **output sample**, gathering rather than scattering. Written\n// this way for a reason: the natural overlap-add scatters frames into a shared\n// buffer, where two frames land on the same sample and the sum needs atomics or\n// a second pass. Reading instead of writing, each output sample is computed by\n// exactly one thread and there is nothing to order. It costs re-deriving the\n// inverse transform once per overlapping frame \u2014 two of them, at 2x overlap.\n//\n// The envelope division is not optional and not an ordinary normalisation: see\n// reference.ts. A periodic Hann at 50% overlap is COLA in w but not in w\xB2, so\n// skipping it is wrong by up to 2x in a way that still sounds like audio.\n\nstruct Params {\n  nFft: u32,\n  hop: u32,\n  bins: u32,\n  frames: u32,\n  outLength: u32,\n  /// floor(nFft/2) when centred, 0 when not. The caller resolves the convention.\n  pad: u32,\n}\n\n@group(0) @binding(0) var<storage, read> real: array<f32>;\n@group(0) @binding(1) var<storage, read> imag: array<f32>;\n@group(0) @binding(2) var<storage, read> win: array<f32>;\n@group(0) @binding(3) var<storage, read_write> out: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst TWO_PI: f32 = 6.28318530717958647692;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let t = global_id.x;\n  if (t >= params.outLength) {\n    return;\n  }\n  let position = i32(t + params.pad);\n  // Even nFft has a Nyquist bin, which stands for itself rather than for a\n  // conjugate pair and so is counted once. Odd nFft has none, and every bin\n  // above DC doubles. torch.fft.irfft splits the same way.\n  let hasNyquist = (params.nFft % 2u) == 0u;\n\n  var numerator: f32 = 0.0;\n  var envelope: f32 = 0.0;\n  for (var frame = 0u; frame < params.frames; frame += 1u) {\n    let n = position - i32(frame * params.hop);\n    // The `n < 0` half of this cannot be caught by a test on this device, and\n    // is written down rather than left to be discovered. Dropping it makes\n    // `u32(n)` wrap to about 4e9; this GPU reads that far past a buffer as\n    // zero, the window value comes back 0, and the frame contributes nothing \u2014\n    // which is accidentally the right answer. WGSL allows an implementation to\n    // clamp the index instead, and a device that clamps would read a real\n    // window value and add a whole frame that does not belong to this sample.\n    if (n < 0 || n >= i32(params.nFft)) {\n      continue;\n    }\n    let base = frame * params.bins;\n    // DC counts once, and its imaginary part is dropped along with Nyquist's.\n    var acc: f32 = real[base];\n    for (var k = 1u; k < params.bins; k += 1u) {\n      let weight = select(2.0, 1.0, hasNyquist && k == params.bins - 1u);\n      // Folded into one turn as an integer, as in the forward kernel.\n      let angle = TWO_PI * (f32((k * u32(n)) % params.nFft) / f32(params.nFft));\n      acc += weight * (real[base + k] * cos(angle) - imag[base + k] * sin(angle));\n    }\n    let w = win[u32(n)];\n    numerator += w * (acc / f32(params.nFft));\n    envelope += w * w;\n  }\n  // No NOLA guard here. A shader cannot raise, and a guard that silently\n  // substituted a value would hand back a waveform for a window that cannot\n  // reconstruct one. The reference refuses those inputs before they get here.\n  out[t] = numerator / envelope;\n}\n";

// ../miocodec/gpu.ts
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
  async istft(real, imag, window2, frames, nFft, hop, pad, outLength) {
    const bins = Math.floor(nFft / 2) + 1;
    const realBuffer = this.upload(real);
    const imagBuffer = this.upload(imag);
    const windowBuffer = this.residentBuffer(window2);
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
    istft: (real, imag, window2, frames, nFft, hop) => (
      // `"same"` resolved here, because the kernel takes a number and has no
      // convention of its own: crop `(nFft - hop) / 2` from each end, which
      // leaves `hop * frames` samples. The reference does the same arithmetic
      // behind the mode name.
      gpu.istft(real, imag, window2, frames, nFft, hop, (nFft - hop) / 2, hop * frames)
    )
  };
}

// ../miocodec/safetensors.ts
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

// ../miocodec/browser.ts
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

// constants.ts
var MAX_NEW_TOKENS = 700;
var MAX_SEQ_LEN = 768;
function maxNewFor(promptLen, maxNew = MAX_NEW_TOKENS) {
  return Math.min(maxNew, MAX_SEQ_LEN - promptLen);
}

// ../../../web-xpu-ops/dist/ops/elementwise/reference.js
var ELEMENTWISE = { add: 0, multiply: 1 };

// kernels.ts
var MATVEC_Q8 = "// matvecQ8 (W8A32 GEMV): out[i] = (sum_k unpack(weight[i, k]) * vector[k]) * scale[i]\n//\n// Layout:\n//   weight: [N, ceil(K/4)] u32, row-major, 4 int8 codes packed per word,\n//           least-significant byte first (word bits 0..7 hold column 4*w+0,\n//           8..15 hold 4*w+1, and so on). A code is stored as its\n//           two's-complement byte, so unpacking sign-extends the low byte\n//           rather than just masking it.\n//   scale:  [N]    f32, one per row (`quantize`'s per-row absmax convention)\n//   vector: [K]    f32, shared by every row\n//   output: [N]    f32\n//\n// Same streaming shape as ops/matvec/wgsl/kernel.wgsl: one workgroup per\n// output row, 256 lanes walking that row with a stride of the workgroup\n// width. The unit of stride here is a *word*, not a column \u2014 lane t reads\n// packed word base + t on each pass, which is 256 consecutive u32 (1024\n// int8 codes) the memory system can serve as a burst, then unpacks its own\n// four codes from the word it already holds instead of issuing four\n// separate byte reads. That is the whole reason to pack in the first place:\n// four columns for the price of one 32-bit transaction.\n//\n// The row scale multiplies the fully-reduced partial sum once, not each of\n// its terms \u2014 algebraically identical since the scale does not depend on\n// the column, and it is `K - 1` fewer multiplies per row.\n\nstruct Params {\n  N: u32,\n  K: u32,\n}\n\n@group(0) @binding(0) var<storage, read> weight: array<u32>;\n@group(0) @binding(1) var<storage, read> scale: array<f32>;\n@group(0) @binding(2) var<storage, read> vector: array<f32>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst WORKGROUP_SIZE: u32 = 256u;\n\nvar<workgroup> shared_sum: array<f32, 256>;\n\n// Sign-extends the byte at `lane` (0..3) of a packed word: bits 0..7 for lane\n// 0, up through 24..31 for lane 3. `extractBits` on a *signed* base carries\n// the sign of the extracted field, which is exactly two's-complement\n// unpacking; `word` only reaches here as a `u32` because storage buffers\n// cannot hold mixed-sign words, so it is bitcast first.\nfn unpack_i8(word: u32, lane: u32) -> f32 {\n  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));\n}\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(workgroup_id) wg_id: vec3<u32>,\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n  let row = wg_id.x;\n  let tid = local_id.x;\n  let words_per_row = (params.K + 3u) / 4u;\n  let row_word_offset = row * words_per_row;\n\n  // Partial dot product, one packed word \u2014 up to four columns \u2014 per lane per\n  // pass. `col < params.K` is the tail guard: when K is not a multiple of 4\n  // the last word's high lane(s) hold padding that must never reach the sum.\n  var partial: f32 = 0.0;\n  for (var word_index = tid; word_index < words_per_row; word_index += WORKGROUP_SIZE) {\n    let word = weight[row_word_offset + word_index];\n    let base_col = word_index * 4u;\n    for (var lane = 0u; lane < 4u; lane += 1u) {\n      let col = base_col + lane;\n      if (col >= params.K) {\n        break;\n      }\n      partial += unpack_i8(word, lane) * vector[col];\n    }\n  }\n\n  shared_sum[tid] = partial;\n  workgroupBarrier();\n\n  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) {\n      shared_sum[tid] += shared_sum[tid + stride];\n    }\n    workgroupBarrier();\n  }\n\n  if (tid == 0u) {\n    output[row] = shared_sum[0] * scale[row];\n  }\n}\n";
var RMSNORM = "// RMSNorm: x_i * w_i / sqrt(mean(x\xB2) + eps)\n//\n// Two-pass within one dispatch:\n//   1. Compute sum of squares (workgroup reduction)\n//   2. Normalize: x_i * w_i * rsqrt(mean_sq + eps)\n//\n// Layout:\n//   input:  [N, D] f32\n//   weight: [G, D] f32 (learnable scale; row n uses group n % G)\n//   output: [N, D] f32\n//\n// G is the extent of the axis immediately left of D \u2014 the fastest varying of\n// the axes flattened into N. For QK-norm's [B, S, H, Dh] that is H, which is\n// why `row % G` is the head index. reference.ts states the layout and why the\n// [B, H, S, Dh] one is refused rather than served wrongly.\n\nstruct Params {\n  N: u32,\n  D: u32,\n  eps: f32,\n  // Group count. Zero means one group: a caller packing the three-word params\n  // this op took before #78 leaves this word zero, and must keep its old\n  // behaviour rather than take an indeterminate `% 0`.\n  //\n  // The `max(params.G, 1u)` below is stated as a contract because **no test can\n  // kill it here**, and that is worth writing down rather than discovering. Rule\n  // 1 asks for a mutation that goes red; this one does not. Measured: dropping\n  // the `max` left every pre-#78 case green, so Tint lowers `x % 0` to `0` on\n  // this device \u2014 the same answer the guard gives. WGSL only promises an\n  // *indeterminate* value, so the guard is what makes the compatibility path a\n  // property of this op rather than of one driver. Deleting it would be green\n  // here and wrong somewhere else.\n  G: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nconst WORKGROUP_SIZE: u32 = 256u;\n\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(workgroup_id) wg_id: vec3<u32>,\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n  let row = wg_id.x;\n  if (row >= params.N) {\n    return;\n  }\n\n  let tid = local_id.x;\n  let row_offset = row * params.D;\n\n  // Pass 1: Sum of squares\n  var local_sum: f32 = 0.0;\n  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {\n    let val = input[row_offset + col];\n    local_sum += val * val;\n  }\n\n  shared_sum[tid] = local_sum;\n  workgroupBarrier();\n\n  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) {\n      shared_sum[tid] += shared_sum[tid + stride];\n    }\n    workgroupBarrier();\n  }\n\n  let rms = inverseSqrt(shared_sum[0] / f32(params.D) + params.eps);\n\n  // Pass 2: Normalize\n  let weight_offset = (row % max(params.G, 1u)) * params.D;\n  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {\n    output[row_offset + col] = input[row_offset + col] * rms * weight[weight_offset + col];\n  }\n}\n";
var ROPE = "// Rotary Position Embeddings (RoPE), with NTK and YaRN context scaling.\n//\n// For each pair (x[2i], x[2i+1]) at position `pos`:\n//   theta = pos * inv_freq(i)\n//   out[2i]   = m * (x[2i] * cos(theta) - x[2i+1] * sin(theta))\n//   out[2i+1] = m * (x[2i] * sin(theta) + x[2i+1] * cos(theta))\n//\n// Layout:\n//   input:  [N, num_heads, head_dim] f32\n//   output: [N, num_heads, head_dim] f32\n//   Dispatched per (token, head, pair)\n//\n// ## One kernel, no variants, no mode switch\n//\n// NTK and YaRN differ from plain RoPE, and from each other, only in how\n// inv_freq(i) is built \u2014 and every ingredient of that beyond `i` itself is\n// constant across the whole tensor. `ops/rope/reference.ts` reduces all three\n// schemes to the five scalars in `Params` below, on the host, once per model.\n//\n// What is left here is one expression that is *every* scheme:\n//\n//   inv_freq = extrapolation + (interpolation - extrapolation) * ramp\n//\n//   - Plain RoPE and NTK arrive with interpolation_factor = 1, which makes\n//     `interpolation - extrapolation` an exact IEEE zero. The ramp then\n//     multiplies zero and inv_freq is the unscaled frequency **bit for bit** \u2014\n//     not merely to within a tolerance. attention_factor = 1 is likewise an\n//     exact multiply. Asking for no scaling really does change nothing.\n//   - NTK is then only a larger `effective_base`. It needed no shader change\n//     at all: the pre-scaling kernel computes NTK correctly if handed the\n//     adjusted base, which is the strongest evidence available that a separate\n//     NTK variant would have been duplication.\n//   - YaRN arrives with interpolation_factor = s and a live ramp.\n//\n// The alternatives cost more and buy nothing. A `mode` uniform with a branch\n// per scheme would move a log, a floor, a ceiling, two clamps and a\n// divide-by-zero guard out of host code that runs once per model and into\n// shader code that runs per element, and each arm would be reachable by\n// exactly one test. Three .wgsl files would keep three copies of the rotation\n// in step for one line of difference each.\n//\n// ## The rotary cache, and what it does when it runs out\n//\n// The angles depend on position and pair alone, so decoding recomputes the same\n// `pow`, `sin` and `cos` for every head, every step. `ropeCache` in the\n// reference precomputes them into `cache` and this kernel reads them instead.\n//\n// A table has a length and decoding runs past it. Of the three things that\n// could happen there, two are unavailable and one is right:\n//\n//   - growing the table needs a host, and there is no host inside a dispatch\n//   - wrapping \u2014 `pos % cache_positions` \u2014 is a real angle for the wrong\n//     position, so it returns a plausible tensor instead of an error, which is\n//     worse than returning nothing\n//   - falling back to computing the angle here is correct, and costs only the\n//     saving\n//\n// So: `pos < cache_positions` reads the table, and everything else computes.\n// The fallback is not a special case \u2014 it is the uncached kernel, unchanged,\n// which is why `cache_positions = 0` is exactly the op this file was before the\n// cache existed. Same degenerate-case argument as the scaling above.\n//\n// `cache` is bound either way, since a binding a shader ignores is dropped by\n// `layout: \"auto\"`. Callers who want no cache bind two floats and pass\n// `cache_positions = 0`; nothing reads them.\n//\n// ## The head range\n//\n// `head_offset` / `head_count` rotate a run of heads and copy the rest. The\n// range is over **heads** \u2014 the `H` of `[N, H, Dh]` \u2014 and not over the channels\n// within a head; `ops/rope/reference.ts` says why that is worth stating twice.\n// `head_offset = 0, head_count = num_heads` is the whole tensor and the same\n// bytes this kernel produced before the range existed, since the branch below\n// is then never taken.\n//\n// The copy is a write, not a skip. Nothing else fills `output`, so a head this\n// dispatch leaves alone reads back as zeros \u2014 a tensor rather than an error,\n// exactly like wrapping the cache would be.\n\nstruct Params {\n  N: u32,          // sequence length\n  num_heads: u32,\n  head_dim: u32,\n  pos_offset: u32, // starting position (for KV-cache continuation)\n  // Positions [0, cache_positions) are in `cache`. 0 means no cache at all.\n  cache_positions: u32,\n\n  // From `ropeFrequencyParams` in the reference, which documents these against\n  // the papers' notation. The short version:\n  effective_base: f32,       // the base actually raised to -2i/D; NTK scales it\n  interpolation_factor: f32, // YaRN's s, dividing the interpolated branch; 1 otherwise\n  ramp_low: f32,             // pair index where YaRN stops extrapolating\n  ramp_high: f32,            // pair index where YaRN is fully interpolated\n  attention_factor: f32,     // YaRN's sqrt(1/t), a gain on cos and sin; 1 otherwise\n\n  // Heads [head_offset, head_offset + head_count) rotate; the rest are copied.\n  // Appended rather than grouped with the geometry above so that every offset\n  // this kernel already read stayed where it was when the range arrived.\n  head_offset: u32,\n  head_count: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n// [cache_positions, head_dim/2, 2] \u2014 cos then sin, with attention_factor\n// already folded in, from `ropeCache`. Built from the same five scalars below,\n// so the table and the fallback are the same rotation by construction.\n@group(0) @binding(1) var<storage, read> cache: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(global_invocation_id) gid: vec3<u32>,\n) {\n  let half_dim = params.head_dim / 2u;\n  let total_pairs = params.N * params.num_heads * half_dim;\n\n  let pair_idx = gid.x;\n  if (pair_idx >= total_pairs) {\n    return;\n  }\n\n  // Decompose linear index into (token, head, dim_pair)\n  let dim_pair = pair_idx % half_dim;\n  let remainder = pair_idx / half_dim;\n  let head = remainder % params.num_heads;\n  let token = remainder / params.num_heads;\n\n  let base_idx = (token * params.num_heads + head) * params.head_dim + dim_pair * 2u;\n  let x0 = input[base_idx];\n  let x1 = input[base_idx + 1u];\n\n  // Heads outside the range are copied, and the copy happens here rather than\n  // being skipped: `output` arrives zeroed, so a head nobody writes comes back\n  // as zeros \u2014 a plausible tensor that takes attention with it, which is the\n  // same failure mode as wrapping the cache above.\n  if (head < params.head_offset || head >= params.head_offset + params.head_count) {\n    output[base_idx]      = x0;\n    output[base_idx + 1u] = x1;\n    return;\n  }\n\n  let pos = token + params.pos_offset;\n\n  var cos_theta: f32;\n  var sin_theta: f32;\n  if (pos < params.cache_positions) {\n    let at = (pos * half_dim + dim_pair) * 2u;\n    cos_theta = cache[at];\n    sin_theta = cache[at + 1u];\n  } else {\n    let freq_exp = -2.0 * f32(dim_pair) / f32(params.head_dim);\n\n    // The reference implementations write these as 1/pos_freqs and\n    // 1/(s*pos_freqs) with pos_freqs = b^(2i/D). Written as a negative exponent\n    // instead, so that the unscaled path is the expression this kernel has\n    // always evaluated.\n    let extrapolation = pow(params.effective_base, freq_exp);\n    let interpolation = extrapolation / params.interpolation_factor;\n\n    // YaRN's gamma(i): 0 below ramp_low \u2014 this pair turns fast enough to have\n    // been seen at every phase during training, so extrapolate \u2014 and 1 above\n    // ramp_high, where the pair has not completed a single rotation and must be\n    // interpolated. Linear between.\n    let ramp = clamp((f32(dim_pair) - params.ramp_low) / (params.ramp_high - params.ramp_low), 0.0, 1.0);\n    let inv_freq = extrapolation + (interpolation - extrapolation) * ramp;\n\n    let theta = f32(pos) * inv_freq;\n\n    // The attention temperature folds into cos/sin, as it does in both\n    // reference implementations. Scaling q and k before the dot product is the\n    // same thing one step later. The table above has it folded in already, for\n    // the same reason and at the same place.\n    cos_theta = cos(theta) * params.attention_factor;\n    sin_theta = sin(theta) * params.attention_factor;\n  }\n\n  output[base_idx]      = x0 * cos_theta - x1 * sin_theta;\n  output[base_idx + 1u] = x0 * sin_theta + x1 * cos_theta;\n}\n";
var GQA_SCORES = "// GQA / MQA, dispatch 1 of 2: probs = softmax(mask(scale * Q @ K_g^T)).\n//\n// Layout:\n//   q:      [B, H,       L, D]  f32, row-major\n//   k:      [B, kv_heads, S, D] f32, row-major   <- fewer heads than q\n//   mask:   [MB, MH, MR, S]     f32, row-major \u2014 the additive attention bias\n//   probs:  [B, H,       L, S]  f32, row-major\n//\n// Structurally this is ops/attention's scores kernel. The only difference is\n// which head of K a query head reads, and that difference is two lines, which is\n// the point: sharing KV heads is an addressing change, not an arithmetic one.\n// Everything about masking, the scale and the softmax is unchanged, and is\n// documented in ops/attention rather than restated here.\n//\n// The dispatch is exactly [L, H, B] \u2014 over the *query* heads, since every query\n// head still gets its own probability row. As in attention, every invocation has\n// real work and no bounds guard is written, because a guard nothing can trip is\n// a guard no test can keep honest (rule 1). Contract: H % kv_heads == 0, checked\n// by the caller (reference.ts throws), so the integer division below is exact.\n\nstruct Params {\n  H: u32,\n  // Key / value heads. H is plain MHA, 1 is MQA, a divisor of H is GQA.\n  kv_heads: u32,\n  L: u32,\n  S: u32,\n  D: u32,\n  scale: f32,\n  causal: u32,\n  query_offset: i32,\n  // The broadcast shape of `mask`, each either 1 or the full dimension; the\n  // last axis is always S. `mask_heads` counts *query* heads, since the mask is\n  // applied after the KV head is chosen (measured against torch \u2014 see\n  // reference.ts).\n  mask_batch: u32,\n  mask_heads: u32,\n  mask_rows: u32,\n  // Issue #117. `S` stays the address stride for `k` (`k_head` below is still\n  // `kv_head * S * D`) \u2014 this only bounds the softmax scan. Defaults to `S`\n  // (append rather than insert, so this field's byte offset does not move\n  // `query_offset`'s \u2014 `llm/engine-q8-resident.ts#GQA_QUERY_OFFSET_BYTE`\n  // copies that offset, rule 2). See reference.ts's `sEff` doc for the\n  // caller's safety contract.\n  s_eff: u32,\n}\n\n@group(0) @binding(0) var<storage, read> q: array<f32>;\n@group(0) @binding(1) var<storage, read> k: array<f32>;\n@group(0) @binding(2) var<storage, read> mask: array<f32>;\n@group(0) @binding(3) var<storage, read_write> probs: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst WORKGROUP_SIZE: u32 = 256u;\n\n// Stands in for the -infinity PyTorch adds as `attn_bias`; WGSL cannot write an\n// infinity literal. See ops/attention/wgsl/scores.wgsl for the measurements that\n// show -FLT_MAX behaves identically for every input this op accepts.\nconst MASKED: f32 = -3.402823e+38;\n\nvar<workgroup> shared_val: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(workgroup_id) wg: vec3<u32>,\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n  let i = wg.x;                                  // query position\n  let batch = wg.z;\n  let head = wg.y;                               // query head\n  let tid = local_id.x;\n\n  let q_head = batch * params.H + head;\n  // Contiguous groups: query heads 0..g-1 share KV head 0, g..2g-1 share KV head\n  // 1, and so on. Strided groups (`head % params.kv_heads`) are the other\n  // reading and are wrong; both agree at kv_heads == H and at kv_heads == 1, so\n  // only the middle cases can tell them apart. The batch stride is kv_heads, not\n  // H \u2014 K has fewer heads per batch, and reusing H here walks past its end.\n  let kv_head = batch * params.kv_heads + head / (params.H / params.kv_heads);\n\n  let q_row = (q_head * params.L + i) * params.D;\n  let k_head = kv_head * params.S * params.D;\n  let p_row = (q_head * params.L + i) * params.S;\n\n  // Where this query row's slice of the bias starts; each axis collapses to 0\n  // when the caller broadcast it. Identical to ops/attention \u2014 deliberately, so\n  // one mask serves both ops.\n  let mb = select(0u, batch, params.mask_batch > 1u);\n  let mh = select(0u, head, params.mask_heads > 1u);\n  let mr = select(0u, i, params.mask_rows > 1u);\n  let m_row = ((mb * params.mask_heads + mh) * params.mask_rows + mr) * params.S;\n\n  // Pass 1: scaled dot products, straight into `probs`. Bounded by `s_eff`,\n  // not `S` \u2014 positions `[s_eff, S)` are never read, not masked-then-summed\n  // (issue #117; see reference.ts's `sEff` doc).\n  var local_max: f32 = MASKED;\n  for (var j = tid; j < params.s_eff; j += WORKGROUP_SIZE) {\n    var value: f32 = MASKED;\n    if (params.causal == 0u || i32(j) <= i32(i) + params.query_offset) {\n      var dot: f32 = 0.0;\n      for (var d: u32 = 0u; d < params.D; d += 1u) {\n        dot += q[q_row + d] * k[k_head + j * params.D + d];\n      }\n      // PyTorch's float `attn_mask`, added rather than selected on. See\n      // ops/attention/wgsl/scores.wgsl.\n      value = dot * params.scale + mask[m_row + j];\n    }\n    probs[p_row + j] = value;\n    local_max = max(local_max, value);\n  }\n\n  shared_val[tid] = local_max;\n  workgroupBarrier();\n  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) {\n      shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);\n    }\n    workgroupBarrier();\n  }\n  let row_max = shared_val[0];\n  workgroupBarrier();\n\n  // Pass 2: exponentiate in place, and sum. The row max comes off first because\n  // exp() overflows f32 at 89 and attention logits reach the hundreds.\n  var local_sum: f32 = 0.0;\n  for (var j = tid; j < params.s_eff; j += WORKGROUP_SIZE) {\n    let e = exp(probs[p_row + j] - row_max);\n    probs[p_row + j] = e;\n    local_sum += e;\n  }\n\n  shared_val[tid] = local_sum;\n  workgroupBarrier();\n  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) {\n      shared_val[tid] += shared_val[tid + stride];\n    }\n    workgroupBarrier();\n  }\n  // Every key masked \u2014 reachable only through `mask`, and returned as zeros\n  // because 1/0 here is +inf and 0 * inf is NaN. The condition is the *sum* and\n  // not the row maximum: `local_max` starts at MASKED so the maximum cannot\n  // report -inf, while the sum is 0 exactly when no score was finite. See\n  // ops/attention/wgsl/scores.wgsl for the torch measurements.\n  let inv_sum = select(1.0 / shared_val[0], 0.0, shared_val[0] == 0.0);\n  workgroupBarrier();\n\n  // Pass 3: normalise. Masked columns are already 0 and stay 0. Columns\n  // `[s_eff, S)` are never written by any pass \u2014 whatever `probs` held there\n  // before this dispatch stays there, and `context.wgsl` must never read it\n  // (it also stops at `s_eff`, given the same params buffer \u2014 see that\n  // file's doc).\n  for (var j = tid; j < params.s_eff; j += WORKGROUP_SIZE) {\n    probs[p_row + j] = probs[p_row + j] * inv_sum;\n  }\n}\n";
var GQA_CONTEXT = "// GQA / MQA, dispatch 2 of 2: output = probs @ V_g.\n//\n// Layout:\n//   probs:  [B, H,        L, S]  f32, row-major \u2014 what scores.wgsl produced\n//   v:      [B, kv_heads, S, Dv] f32, row-major   <- fewer heads than probs\n//   output: [B, H,        L, Dv] f32, row-major\n//\n// ops/attention's context kernel with the same one change scores.wgsl makes:\n// which head of V a query head reads. Nothing here knows the mask exists \u2014\n// masked columns arrive as exactly 0 from the softmax \u2014 which is why sharing KV\n// heads and choosing a causal convention stay independent of each other.\n//\n// Dispatch is exactly [L, H, B], over the query heads, so no bounds guard is\n// reachable and none is written (rule 1).\n\nstruct Params {\n  H: u32,\n  kv_heads: u32,\n  L: u32,\n  S: u32,\n  Dv: u32,\n  // Issue #117, same field and same contract as scores.wgsl's: bounds the\n  // `probs @ V` sum, `S` stays `v`'s per-head stride. Must be given the same\n  // value `scores.wgsl` was \u2014 `probs` columns past it were never written by\n  // that dispatch (see scores.wgsl's doc), so reading further here would read\n  // whatever this buffer held before. Default `S` (appended, not inserted \u2014\n  // no existing field's offset moves).\n  s_eff: u32,\n}\n\n@group(0) @binding(0) var<storage, read> probs: array<f32>;\n@group(0) @binding(1) var<storage, read> v: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nconst WORKGROUP_SIZE: u32 = 256u;\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(workgroup_id) wg: vec3<u32>,\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n  let i = wg.x;                                  // query position\n  let batch = wg.z;\n  let head = wg.y;                               // query head\n\n  let q_head = batch * params.H + head;\n  // Same mapping as scores.wgsl, and it has to stay the same: K and V are\n  // indexed by the same KV head, so a disagreement between the two dispatches\n  // would attend to one head's keys and read another head's values.\n  let kv_head = batch * params.kv_heads + head / (params.H / params.kv_heads);\n\n  let p_row = (q_head * params.L + i) * params.S;\n  let v_head = kv_head * params.S * params.Dv;\n  let o_row = (q_head * params.L + i) * params.Dv;\n\n  // One output channel per invocation, striding when Dv exceeds the workgroup.\n  for (var c = local_id.x; c < params.Dv; c += WORKGROUP_SIZE) {\n    var acc: f32 = 0.0;\n    for (var j: u32 = 0u; j < params.s_eff; j += 1u) {\n      acc += probs[p_row + j] * v[v_head + j * params.Dv + c];\n    }\n    output[o_row + c] = acc;\n  }\n}\n";
var ACTIVATION2 = '// Elementwise activations.\n//\n// Every kind here is a pure function of one element: no layout, no second\n// buffer, `N` is all the kernel is told. Snake looks like it belongs and does\n// not \u2014 its alpha is a trained per-channel tensor, so it needs a storage\n// binding and a channel stride, and it lives in `ops/snake`. See\n// `ops/snake/reference.ts` for that decision.\n//\n//   0 ReLU\xB2      max(0, x)\xB2                     BitNet 2B-4T FFN\n//   1 SiLU       x \xB7 sigmoid(x)                 torch.nn.SiLU\n//   2 ELU        x if x > 0 else \u03B1(e\u02E3 - 1)      torch.nn.ELU, \u03B1 default 1.0\n//   3 Tanh       tanh(x)                        torch.tanh\n//   4 GELU       0.5x(1 + erf(x/\u221A2))            torch gelu, approximate="none"\n//   5 GELU tanh  0.5x(1 + tanh(\u221A(2/\u03C0)(x + 0.044715x\xB3)))   approximate="tanh"\n//\n// 4 and 5 are different functions, not two spellings of one: they part company\n// by 4.73e-4 at x = 2.699. torch\'s default is 4, so this op\'s default is 4.\n//\n// Layout:\n//   input:  [N] f32\n//   output: [N] f32\n\nstruct Params {\n  N: u32,\n  activation_type: u32,\n  alpha: f32,  // ELU\'s negative-branch scale; ignored by every other kind\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n/// erfc for u >= 0: Abramowitz & Stegun 7.1.26, whose stated error bound is\n/// 1.5e-7 \u2014 the ulp of 1 in f32, so it is as close as this type gets. WGSL has\n/// no erf and no erfc, so the exact GELU has to carry one.\nfn erfc_nonneg(u: f32) -> f32 {\n  let t = 1.0 / (1.0 + 0.3275911 * u);\n  let poly = t * (0.254829592\n      + t * (-0.284496736\n      + t * (1.421413741\n      + t * (-1.453152027\n      + t * 1.061405429))));\n  return poly * exp(-u * u);\n}\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(global_invocation_id) gid: vec3<u32>,\n) {\n  let idx = gid.x;\n  if (idx >= params.N) {\n    return;\n  }\n\n  let x = input[idx];\n  var y: f32;\n\n  switch (params.activation_type) {\n    case 0u: {\n      // ReLU\xB2: max(0, x)\xB2\n      let relu_x = max(0.0, x);\n      y = relu_x * relu_x;\n    }\n    case 1u: {\n      // SiLU: x * sigmoid(x)\n      y = x / (1.0 + exp(-x));\n    }\n    case 2u: {\n      // ELU. `x > 0` as torch writes it; at x = 0 both arms are 0 anyway.\n      if (x > 0.0) {\n        y = x;\n      } else {\n        y = params.alpha * (exp(x) - 1.0);\n      }\n    }\n    case 3u: {\n      y = tanh(x);\n    }\n    case 4u: {\n      // GELU, exact. Written through erfc rather than erf so neither side\n      // cancels: for x >= 0, 1 + erf(u) is 2 - erfc(u), and for x < 0 it is\n      // erfc(|u|) outright, rather than 1 + (1 - erfc) \u2014 which for x = -8 is a\n      // subtraction between two f32 numbers a hair either side of 1, and\n      // returns exactly zero where the answer is -3e-14.\n      //\n      // Stated as a contract, not as something these tests check: the naive\n      // arrangement was written in and the suite stayed green (measured). It\n      // cannot catch it, for two compounding reasons \u2014 the harness passes an\n      // element on absolute agreement below 1e-6, and every value this affects\n      // is far below that; and the f64 reference computes `1 + erf` and so\n      // cancels in the same place. Kept because it is free and right, not\n      // because a red test demanded it.\n      let u = x * 0.70710678118654752;\n      let q = erfc_nonneg(abs(u));\n      if (x >= 0.0) {\n        y = x * (1.0 - 0.5 * q);\n      } else {\n        y = 0.5 * x * q;\n      }\n    }\n    default: {\n      // GELU, tanh approximation. \u221A(2/\u03C0) = 0.7978845608028654.\n      y = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));\n    }\n  }\n\n  output[idx] = y;\n}\n';
var ELEMENTWISE2 = "// Elementwise operations: add, multiply\n//\n// Used for residual connections and gating.\n//\n// Layout:\n//   a:      [N] f32\n//   b:      [N] f32\n//   output: [N] f32\n\nstruct Params {\n  N: u32,\n  op: u32,  // 0 = add, 1 = multiply\n}\n\n@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(global_invocation_id) gid: vec3<u32>,\n) {\n  let idx = gid.x;\n  if (idx >= params.N) {\n    return;\n  }\n\n  if (params.op == 0u) {\n    output[idx] = a[idx] + b[idx];\n  } else {\n    output[idx] = a[idx] * b[idx];\n  }\n}\n";

// ../../../web-xpu-ops/dist/ops/matvec/reference.js
function packQ8({ codes, N, K }) {
  const wordsPerRow = Math.ceil(K / 4);
  const packed = new Uint32Array(N * wordsPerRow);
  for (let row = 0; row < N; row += 1) {
    for (let word = 0; word < wordsPerRow; word += 1) {
      let bits = 0;
      for (let lane = 0; lane < 4; lane += 1) {
        const col = word * 4 + lane;
        if (col >= K)
          break;
        const byte = codes[row * K + col] & 255;
        bits |= byte << lane * 8;
      }
      packed[row * wordsPerRow + word] = bits >>> 0;
    }
  }
  return packed;
}

// weights-q8.ts
function checkFile(label, meta, buffer, sha256) {
  if (buffer.byteLength !== meta.bytes) {
    throw new Error(
      `${label} (${meta.name}) is ${buffer.byteLength} bytes, manifest says ${meta.bytes}`
    );
  }
  if (!sha256) return;
  const digest = sha256(new Uint8Array(buffer));
  if (digest !== meta.sha256) {
    throw new Error(
      `${label} (${meta.name}) sha256 ${digest.slice(0, 12)}... does not match the manifest's ${meta.sha256.slice(0, 12)}... \u2014 a stale or truncated artifact, regenerate rather than guessing`
    );
  }
}
function loadWeightsQ8({ manifest, codes, scales, norms, sha256 }) {
  if (manifest.ropePermuted !== true) {
    throw new Error("manifest.ropePermuted is not true; this loader only accepts permuted artifacts");
  }
  checkFile("codes", manifest.files.codes, codes, sha256);
  checkFile("scales", manifest.files.scales, scales, sha256);
  checkFile("norms", manifest.files.norms, norms, sha256);
  const quantByName = /* @__PURE__ */ new Map();
  const normByName = /* @__PURE__ */ new Map();
  for (const entry of manifest.tensors) {
    if (entry.kind === "quant") quantByName.set(entry.name, entry);
    else normByName.set(entry.name, entry);
  }
  const getQuant = (name) => {
    const entry = quantByName.get(name);
    if (!entry) throw new Error(`no quantized tensor named ${JSON.stringify(name)} in manifest`);
    const { rows, cols } = entry;
    if (entry.codesBytes !== rows * cols || entry.scaleBytes !== rows * 4) {
      throw new Error(`${name}: byte counts disagree with rows=${rows} cols=${cols}`);
    }
    const raw = new Int8Array(codes, entry.codesOffset, rows * cols);
    const packed = packQ8({ codes: raw, N: rows, K: cols });
    const scale = new Float32Array(scales, entry.scaleOffset, rows).slice();
    return { packed, scale, rows, cols };
  };
  const getNorm = (name) => {
    const entry = normByName.get(name);
    if (!entry) throw new Error(`no norm tensor named ${JSON.stringify(name)} in manifest`);
    return new Float32Array(norms, entry.offset, entry.cols).slice();
  };
  const perLayer = [];
  for (let i = 0; i < manifest.config.numLayers; i += 1) {
    perLayer.push({
      attnNorm: getNorm(`layers.${i}.attnNorm`),
      wq: getQuant(`layers.${i}.wq`),
      wk: getQuant(`layers.${i}.wk`),
      wv: getQuant(`layers.${i}.wv`),
      wo: getQuant(`layers.${i}.wo`),
      ffnNorm: getNorm(`layers.${i}.ffnNorm`),
      wGate: getQuant(`layers.${i}.wGate`),
      wUp: getQuant(`layers.${i}.wUp`),
      wDown: getQuant(`layers.${i}.wDown`),
      qNorm: getNorm(`layers.${i}.qNorm`),
      kNorm: getNorm(`layers.${i}.kNorm`),
      qNormRaw: getNorm(`layers.${i}.qNormRaw`),
      kNormRaw: getNorm(`layers.${i}.kNormRaw`)
    });
  }
  return {
    config: manifest.config,
    embedTokens: getQuant("embedTokens"),
    perLayer,
    finalNorm: getNorm("finalNorm")
  };
}
function signExtend(byte) {
  return byte >= 128 ? byte - 256 : byte;
}
function unpackRowCodes(weight, row) {
  if (row < 0 || row >= weight.rows) throw new Error(`row ${row} out of [0, ${weight.rows})`);
  const wordsPerRow = Math.ceil(weight.cols / 4);
  const base = row * wordsPerRow;
  const out = new Int32Array(weight.cols);
  for (let col = 0; col < weight.cols; col += 1) {
    const word = weight.packed[base + (col >> 2)];
    out[col] = signExtend(word >>> (col & 3) * 8 & 255);
  }
  return out;
}
function dequantizeRow(weight, row) {
  const codes = unpackRowCodes(weight, row);
  const s = weight.scale[row];
  const out = new Float32Array(weight.cols);
  for (let col = 0; col < weight.cols; col += 1) out[col] = codes[col] * s;
  return out;
}
function gatherDequantRow(table, id) {
  if (id < 0 || id >= table.rows) return new Float32Array(table.cols);
  return dequantizeRow(table, id);
}

// gpu-engine.ts
var MAX_WORKGROUPS_PER_DISPATCH = 65535;
var ROPE_POS_OFFSET_BYTE = 3 * 4;
var GQA_QUERY_OFFSET_BYTE = 7 * 4;
var GQA_SCORES_S_EFF_BYTE = 11 * 4;
var GQA_CONTEXT_S_EFF_BYTE = 5 * 4;
function packFields(fields) {
  const buffer = new ArrayBuffer(Math.max(16, fields.length * 4));
  const view = new DataView(buffer);
  fields.forEach(([kind, value], index) => {
    if (kind === "f32") view.setFloat32(index * 4, value, true);
    else if (kind === "i32") view.setInt32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return buffer;
}
async function createGpuEngine(device, weights, opts) {
  const cfg = weights.config;
  const { numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, ropeTheta, rmsNormEps } = cfg;
  const { maxSeqLen } = opts;
  const qDim = numHeads * headDim;
  const kvDim = numKvHeads * headDim;
  const stats = {
    buffersCreated: 0,
    submits: 0,
    dispatchesPerStep: 0,
    copiesPerStep: 0,
    readbackBytesPerStep: vocabSize * 4
  };
  const pipelines = /* @__PURE__ */ new Map();
  const modules = /* @__PURE__ */ new Map();
  const trackedBuffers = [];
  function createStorageBuffer(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    stats.buffersCreated += 1;
    const buffer = device.createBuffer({ size: Math.max(4, bytes), usage });
    trackedBuffers.push(buffer);
    return buffer;
  }
  function upload(buffer, offset, data) {
    device.queue.writeBuffer(buffer, offset, data.buffer, data.byteOffset, data.byteLength);
  }
  function uniformOf(fields) {
    stats.buffersCreated += 1;
    const data = packFields(fields);
    const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    trackedBuffers.push(buffer);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }
  async function pipelineFor(code) {
    const cached = pipelines.get(code);
    if (cached) return cached;
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        throw new Error(`shader failed to compile
${errors.map((m) => `${m.lineNum}:${m.linePos}: ${m.message}`).join("\n")}`);
      }
      modules.set(code, module);
    }
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`pipeline is not valid: ${invalid.message}`);
    pipelines.set(code, pipeline);
    return pipeline;
  }
  async function bindGroup(pipeline, buffers) {
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } }))
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`bind group is not valid: ${invalid.message}`);
    return group;
  }
  function uploadPacked(w) {
    const weightBuf = createStorageBuffer(w.packed.byteLength);
    upload(weightBuf, 0, w.packed);
    const scaleBuf = createStorageBuffer(w.scale.byteLength);
    upload(scaleBuf, 0, w.scale);
    return { weightBuf, scaleBuf };
  }
  const [matvecPipeline, rmsnormPipeline, ropePipeline, scoresPipeline, contextPipeline, activationPipeline, elementwisePipeline] = await Promise.all([
    pipelineFor(MATVEC_Q8),
    pipelineFor(RMSNORM),
    pipelineFor(ROPE),
    pipelineFor(GQA_SCORES),
    pipelineFor(GQA_CONTEXT),
    pipelineFor(ACTIVATION2),
    pipelineFor(ELEMENTWISE2)
  ]);
  const hiddenA = createStorageBuffer(hiddenSize * 4);
  const hiddenB = createStorageBuffer(hiddenSize * 4);
  const normedBuf = createStorageBuffer(hiddenSize * 4);
  const normed2Buf = createStorageBuffer(hiddenSize * 4);
  const qProjBuf = createStorageBuffer(qDim * 4);
  const kProjBuf = createStorageBuffer(kvDim * 4);
  const vProjBuf = createStorageBuffer(kvDim * 4);
  const qNormedBuf = createStorageBuffer(qDim * 4);
  const kNormedBuf = createStorageBuffer(kvDim * 4);
  const qRopedBuf = createStorageBuffer(qDim * 4);
  const kRopedBuf = createStorageBuffer(kvDim * 4);
  const attnOutBuf = createStorageBuffer(qDim * 4);
  const projOutBuf = createStorageBuffer(hiddenSize * 4);
  const gateOutBuf = createStorageBuffer(ffnHidden * 4);
  const upOutBuf = createStorageBuffer(ffnHidden * 4);
  const gateActBuf = createStorageBuffer(ffnHidden * 4);
  const gatedBuf = createStorageBuffer(ffnHidden * 4);
  const downOutBuf = createStorageBuffer(hiddenSize * 4);
  const finalNormedBuf = createStorageBuffer(hiddenSize * 4);
  const dummyCacheBuf = createStorageBuffer(8);
  const maskBuf = createStorageBuffer(maxSeqLen * 4);
  const probsBuf = createStorageBuffer(numHeads * maxSeqLen * 4);
  const hiddenNormUniform = uniformOf([["u32", 1], ["u32", hiddenSize], ["f32", rmsNormEps], ["u32", 1]]);
  const qNormUniform = uniformOf([["u32", numHeads], ["u32", headDim], ["f32", rmsNormEps], ["u32", 1]]);
  const kNormUniform = uniformOf([["u32", numKvHeads], ["u32", headDim], ["f32", rmsNormEps], ["u32", 1]]);
  const qUniform = uniformOf([["u32", qDim], ["u32", hiddenSize]]);
  const kUniform = uniformOf([["u32", kvDim], ["u32", hiddenSize]]);
  const vUniform = uniformOf([["u32", kvDim], ["u32", hiddenSize]]);
  const oUniform = uniformOf([["u32", hiddenSize], ["u32", qDim]]);
  const gateUniform = uniformOf([["u32", ffnHidden], ["u32", hiddenSize]]);
  const upUniform = uniformOf([["u32", ffnHidden], ["u32", hiddenSize]]);
  const downUniform = uniformOf([["u32", hiddenSize], ["u32", ffnHidden]]);
  const siluUniform = uniformOf([["u32", ffnHidden], ["u32", ACTIVATION.silu], ["f32", 1]]);
  const addUniform = uniformOf([["u32", hiddenSize], ["u32", ELEMENTWISE.add]]);
  const mulUniform = uniformOf([["u32", ffnHidden], ["u32", ELEMENTWISE.multiply]]);
  const ropeQUniform = uniformOf([
    ["u32", 1],
    ["u32", numHeads],
    ["u32", headDim],
    ["u32", 0],
    ["u32", 0],
    ["f32", ropeTheta],
    ["f32", 1],
    ["f32", 0],
    ["f32", 1],
    ["f32", 1],
    ["u32", 0],
    ["u32", numHeads]
  ]);
  const ropeKUniform = uniformOf([
    ["u32", 1],
    ["u32", numKvHeads],
    ["u32", headDim],
    ["u32", 0],
    ["u32", 0],
    ["f32", ropeTheta],
    ["f32", 1],
    ["f32", 0],
    ["f32", 1],
    ["f32", 1],
    ["u32", 0],
    ["u32", numKvHeads]
  ]);
  const scoresUniform = uniformOf([
    ["u32", numHeads],
    ["u32", numKvHeads],
    ["u32", 1],
    ["u32", maxSeqLen],
    ["u32", headDim],
    ["f32", 1 / Math.sqrt(headDim)],
    ["u32", 1],
    ["i32", 0],
    ["u32", 1],
    ["u32", 1],
    ["u32", 1],
    ["u32", maxSeqLen]
  ]);
  const contextUniform = uniformOf([
    ["u32", numHeads],
    ["u32", numKvHeads],
    ["u32", 1],
    ["u32", maxSeqLen],
    ["u32", headDim],
    ["u32", maxSeqLen]
  ]);
  const ropeQGroup = await bindGroup(ropePipeline, [qNormedBuf, dummyCacheBuf, qRopedBuf, ropeQUniform]);
  const ropeKGroup = await bindGroup(ropePipeline, [kNormedBuf, dummyCacheBuf, kRopedBuf, ropeKUniform]);
  const add1Group = await bindGroup(elementwisePipeline, [hiddenA, projOutBuf, hiddenB, addUniform]);
  const siluGroup = await bindGroup(activationPipeline, [gateOutBuf, gateActBuf, siluUniform]);
  const mulGroup = await bindGroup(elementwisePipeline, [gateActBuf, upOutBuf, gatedBuf, mulUniform]);
  const add2Group = await bindGroup(elementwisePipeline, [hiddenB, downOutBuf, hiddenA, addUniform]);
  async function projectionGroup(w, uniform, vectorBuf, outBuf) {
    const { weightBuf, scaleBuf } = uploadPacked(w);
    return bindGroup(matvecPipeline, [weightBuf, scaleBuf, vectorBuf, outBuf, uniform]);
  }
  const layers = [];
  for (const lw of weights.perLayer) {
    const attnNormBuf = createStorageBuffer(lw.attnNorm.byteLength);
    upload(attnNormBuf, 0, lw.attnNorm);
    const ffnNormBuf = createStorageBuffer(lw.ffnNorm.byteLength);
    upload(ffnNormBuf, 0, lw.ffnNorm);
    const qGammaBuf = createStorageBuffer(lw.qNorm.byteLength);
    upload(qGammaBuf, 0, lw.qNorm);
    const kGammaBuf = createStorageBuffer(lw.kNorm.byteLength);
    upload(kGammaBuf, 0, lw.kNorm);
    const kCacheBuf = createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4);
    const vCacheBuf = createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4);
    layers.push({
      kCacheBuf,
      vCacheBuf,
      attnNormGroup: await bindGroup(rmsnormPipeline, [hiddenA, attnNormBuf, normedBuf, hiddenNormUniform]),
      ffnNormGroup: await bindGroup(rmsnormPipeline, [hiddenB, ffnNormBuf, normed2Buf, hiddenNormUniform]),
      wqGroup: await projectionGroup(lw.wq, qUniform, normedBuf, qProjBuf),
      wkGroup: await projectionGroup(lw.wk, kUniform, normedBuf, kProjBuf),
      wvGroup: await projectionGroup(lw.wv, vUniform, normedBuf, vProjBuf),
      qNormGroup: await bindGroup(rmsnormPipeline, [qProjBuf, qGammaBuf, qNormedBuf, qNormUniform]),
      kNormGroup: await bindGroup(rmsnormPipeline, [kProjBuf, kGammaBuf, kNormedBuf, kNormUniform]),
      scoresGroup: await bindGroup(scoresPipeline, [qRopedBuf, kCacheBuf, maskBuf, probsBuf, scoresUniform]),
      contextGroup: await bindGroup(contextPipeline, [probsBuf, vCacheBuf, attnOutBuf, contextUniform]),
      woGroup: await projectionGroup(lw.wo, oUniform, attnOutBuf, projOutBuf),
      gateGroup: await projectionGroup(lw.wGate, gateUniform, normed2Buf, gateOutBuf),
      upGroup: await projectionGroup(lw.wUp, upUniform, normed2Buf, upOutBuf),
      downGroup: await projectionGroup(lw.wDown, downUniform, gatedBuf, downOutBuf)
    });
  }
  const finalNormBuf = createStorageBuffer(weights.finalNorm.byteLength);
  upload(finalNormBuf, 0, weights.finalNorm);
  const finalNormGroup = await bindGroup(rmsnormPipeline, [hiddenA, finalNormBuf, finalNormedBuf, hiddenNormUniform]);
  const wordsPerRow = Math.ceil(hiddenSize / 4);
  const lmHeadChunks = [];
  for (let rowStart = 0; rowStart < vocabSize; rowStart += MAX_WORKGROUPS_PER_DISPATCH) {
    const rowCount = Math.min(MAX_WORKGROUPS_PER_DISPATCH, vocabSize - rowStart);
    const chunkUniform = uniformOf([["u32", rowCount], ["u32", hiddenSize]]);
    const outBuf = createStorageBuffer(rowCount * 4);
    const staging = createStorageBuffer(rowCount * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const group = await projectionGroup(
      {
        packed: weights.embedTokens.packed.subarray(rowStart * wordsPerRow, (rowStart + rowCount) * wordsPerRow),
        scale: weights.embedTokens.scale.subarray(rowStart, rowStart + rowCount),
        rows: rowCount,
        cols: hiddenSize
      },
      chunkUniform,
      finalNormedBuf,
      outBuf
    );
    lmHeadChunks.push({ rowCount, group, outBuf, staging });
  }
  const wg256 = (n) => Math.ceil(n / 256);
  let position = 0;
  let firstStepCounted = false;
  async function decodeStep(tokenId, opts2) {
    const skipLogits = opts2?.skipLogits ?? false;
    const at = position;
    if (at + 1 > maxSeqLen) {
      throw new Error(`decodeStep: position ${at + 1} exceeds maxSeqLen=${maxSeqLen}`);
    }
    const embedRow = gatherDequantRow(weights.embedTokens, tokenId);
    upload(hiddenA, 0, embedRow);
    upload(ropeQUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    upload(ropeKUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    upload(scoresUniform, GQA_QUERY_OFFSET_BYTE, new Int32Array([at]));
    const sEff = new Uint32Array([at + 1]);
    upload(scoresUniform, GQA_SCORES_S_EFF_BYTE, sEff);
    upload(contextUniform, GQA_CONTEXT_S_EFF_BYTE, sEff);
    const ops = [];
    const dispatch = (pipeline, group, workgroups) => ops.push({ kind: "dispatch", pipeline, bindGroup: group, workgroups });
    const copy = (src, srcOffset, dst, dstOffset, size) => ops.push({ kind: "copy", src, srcOffset, dst, dstOffset, size });
    for (const layer of layers) {
      dispatch(rmsnormPipeline, layer.attnNormGroup, [1]);
      dispatch(matvecPipeline, layer.wqGroup, [qDim]);
      dispatch(matvecPipeline, layer.wkGroup, [kvDim]);
      dispatch(matvecPipeline, layer.wvGroup, [kvDim]);
      dispatch(rmsnormPipeline, layer.qNormGroup, [numHeads]);
      dispatch(rmsnormPipeline, layer.kNormGroup, [numKvHeads]);
      dispatch(ropePipeline, ropeQGroup, [wg256(qDim / 2)]);
      dispatch(ropePipeline, ropeKGroup, [wg256(kvDim / 2)]);
      for (let h = 0; h < numKvHeads; h += 1) {
        copy(kRopedBuf, h * headDim * 4, layer.kCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
        copy(vProjBuf, h * headDim * 4, layer.vCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
      }
      dispatch(scoresPipeline, layer.scoresGroup, [1, numHeads, 1]);
      dispatch(contextPipeline, layer.contextGroup, [1, numHeads, 1]);
      dispatch(matvecPipeline, layer.woGroup, [hiddenSize]);
      dispatch(elementwisePipeline, add1Group, [wg256(hiddenSize)]);
      dispatch(rmsnormPipeline, layer.ffnNormGroup, [1]);
      dispatch(matvecPipeline, layer.gateGroup, [ffnHidden]);
      dispatch(matvecPipeline, layer.upGroup, [ffnHidden]);
      dispatch(activationPipeline, siluGroup, [wg256(ffnHidden)]);
      dispatch(elementwisePipeline, mulGroup, [wg256(ffnHidden)]);
      dispatch(matvecPipeline, layer.downGroup, [hiddenSize]);
      dispatch(elementwisePipeline, add2Group, [wg256(hiddenSize)]);
    }
    if (!skipLogits) {
      dispatch(rmsnormPipeline, finalNormGroup, [1]);
      for (const chunk of lmHeadChunks) dispatch(matvecPipeline, chunk.group, [chunk.rowCount]);
    }
    if (!firstStepCounted && !skipLogits) {
      firstStepCounted = true;
      stats.dispatchesPerStep = ops.filter((op) => op.kind === "dispatch").length;
      stats.copiesPerStep = ops.filter((op) => op.kind === "copy").length;
    }
    const encoder = device.createCommandEncoder();
    let pass = null;
    const endPass = () => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };
    for (const op of ops) {
      if (op.kind === "dispatch") {
        if (!pass) pass = encoder.beginComputePass();
        pass.setPipeline(op.pipeline);
        pass.setBindGroup(0, op.bindGroup);
        pass.dispatchWorkgroups(...op.workgroups);
      } else {
        endPass();
        encoder.copyBufferToBuffer(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
      }
    }
    endPass();
    if (!skipLogits) {
      for (const chunk of lmHeadChunks) {
        encoder.copyBufferToBuffer(chunk.outBuf, 0, chunk.staging, 0, chunk.rowCount * 4);
      }
    }
    device.queue.submit([encoder.finish()]);
    stats.submits += 1;
    if (skipLogits) {
      position += 1;
      return null;
    }
    const logits = new Float32Array(vocabSize);
    let offset = 0;
    for (const chunk of lmHeadChunks) {
      await chunk.staging.mapAsync(GPUMapMode.READ);
      logits.set(new Float32Array(chunk.staging.getMappedRange().slice(0)), offset);
      chunk.staging.unmap();
      offset += chunk.rowCount;
    }
    position += 1;
    return logits;
  }
  return {
    decodeStep,
    get position() {
      return position;
    },
    reset() {
      position = 0;
    },
    stats,
    destroy() {
      for (const buf of trackedBuffers) buf.destroy();
    }
  };
}

// text.ts
var REPLACE = [
  [/\t/g, ""],
  [/\[n\]/g, ""],
  [/ /g, ""],
  [/　/g, ""],
  // 　 full-width space
  [/[;▼♀♂《》≪≫①②③④⑤⑥]/g, ""],
  // The reference's dash/bar class, escape for escape (˗ ‐‑‒–—― ⁃ − ⎯ ⏤ ─ ━ ⸺ ⸻).
  [/[˗‐-―⁃−⎯⏤─━⸺⸻]/g, ""],
  [/[～〜]/g, "\u30FC"],
  // ～ and 〜 both become the long-vowel bar
  [/？/g, "?"],
  [/！/g, "!"],
  [/[●◯〇]/g, "\u25CB"],
  [/♥/g, "\u2661"]
];
var FULLWIDTH_ALNUM = /[Ａ-Ｚａ-ｚ０-９]/g;
var HALFWIDTH_KATAKANA = "\uFF66\uFF67\uFF68\uFF69\uFF6A\uFF6B\uFF6C\uFF6D\uFF6E\uFF6F\uFF70\uFF71\uFF72\uFF73\uFF74\uFF75\uFF76\uFF77\uFF78\uFF79\uFF7A\uFF7B\uFF7C\uFF7D\uFF7E\uFF7F\uFF80\uFF81\uFF82\uFF83\uFF84\uFF85\uFF86\uFF87\uFF88\uFF89\uFF8A\uFF8B\uFF8C\uFF8D\uFF8E\uFF8F\uFF90\uFF91\uFF92\uFF93\uFF94\uFF95\uFF96\uFF97\uFF98\uFF99\uFF9A\uFF9B\uFF9C\uFF9D";
var FULLWIDTH_KATAKANA = "\u30F2\u30A1\u30A3\u30A5\u30A7\u30A9\u30E3\u30E5\u30E7\u30C3\u30FC\u30A2\u30A4\u30A6\u30A8\u30AA\u30AB\u30AD\u30AF\u30B1\u30B3\u30B5\u30B7\u30B9\u30BB\u30BD\u30BF\u30C1\u30C4\u30C6\u30C8\u30CA\u30CB\u30CC\u30CD\u30CE\u30CF\u30D2\u30D5\u30D8\u30DB\u30DE\u30DF\u30E0\u30E1\u30E2\u30E4\u30E6\u30E8\u30E9\u30EA\u30EB\u30EC\u30ED\u30EF\u30F3";
var KATAKANA_FOLD = /* @__PURE__ */ new Map();
for (let i = 0; i < HALFWIDTH_KATAKANA.length; i += 1) {
  KATAKANA_FOLD.set(HALFWIDTH_KATAKANA[i], FULLWIDTH_KATAKANA[i]);
}
var BRACKET_PAIRS = [
  ["\u300C", "\u300D"],
  ["\u300E", "\u300F"],
  ["\uFF08", "\uFF09"],
  ["\u3010", "\u3011"],
  ["(", ")"]
];
function normalizeText(text) {
  for (const [pattern, replacement] of REPLACE) {
    text = text.replace(pattern, replacement);
  }
  text = text.replace(FULLWIDTH_ALNUM, (ch) => String.fromCharCode(ch.charCodeAt(0) - 65248));
  text = text.replace(/[ｦ-ﾝ]/g, (ch) => KATAKANA_FOLD.get(ch) ?? ch);
  text = text.replace(/…{3,}/g, "\u2026\u2026");
  for (const [open, close] of BRACKET_PAIRS) {
    if (text.startsWith(open) && text.endsWith(close)) {
      text = text.slice(1, -1);
    }
  }
  if (text.endsWith("\u3002") || text.endsWith("\u3001")) {
    text = text.replace(/[。、]+$/, "");
  }
  return text;
}

// tokenizer.ts
var SPEECH_TOKEN_BASE = 151669;
var SPEECH_TOKEN_COUNT = 12800;
function speechIndexOf(id) {
  if (!Number.isInteger(id) || id < SPEECH_TOKEN_BASE || id >= SPEECH_TOKEN_BASE + SPEECH_TOKEN_COUNT) {
    return null;
  }
  return id - SPEECH_TOKEN_BASE;
}
function buildByteMaps() {
  const bytes = [];
  for (let b = 33; b <= 126; b += 1) bytes.push(b);
  for (let b = 161; b <= 172; b += 1) bytes.push(b);
  for (let b = 174; b <= 255; b += 1) bytes.push(b);
  const present = new Set(bytes);
  const byteToChar = new Array(256);
  const charToByte = /* @__PURE__ */ new Map();
  let displaced = 0;
  for (let b = 0; b < 256; b += 1) {
    const code = present.has(b) ? b : 256 + displaced++;
    const ch = String.fromCharCode(code);
    byteToChar[b] = ch;
    charToByte.set(ch, b);
  }
  return { byteToChar, charToByte };
}
var { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteMaps();
var PRE_TOKENIZE = new RegExp(
  "(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
  "giu"
);
var SPEECH_TOKEN_RE = /^<\|s_(\d+)\|>$/;
var BpeTokenizer = class {
  vocab;
  pieces;
  ranks;
  /** Added tokens *except* the 12,800 `<|s_n|>` forms. */
  addedByContent;
  addedById;
  /** Alternation over the non-speech added tokens plus one `<|s_n|>` branch. */
  addedSplitter;
  bpeCache = /* @__PURE__ */ new Map();
  constructor(json) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.pieces = /* @__PURE__ */ new Map();
    for (const [piece, id] of this.vocab) this.pieces.set(id, piece);
    this.ranks = /* @__PURE__ */ new Map();
    json.model.merges.forEach((merge, rank) => {
      const [a, b] = typeof merge === "string" ? splitMergeString(merge) : merge;
      this.ranks.set(`${a} ${b}`, rank);
    });
    this.addedByContent = /* @__PURE__ */ new Map();
    this.addedById = /* @__PURE__ */ new Map();
    const named = [];
    for (const { id, content } of json.added_tokens) {
      this.addedById.set(id, content);
      const speech = SPEECH_TOKEN_RE.exec(content);
      if (speech && speechIndexOf(id) === Number(speech[1])) continue;
      this.addedByContent.set(content, id);
      named.push(content);
    }
    named.sort((a, b) => b.length - a.length);
    const escaped = named.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.addedSplitter = new RegExp(`${escaped.join("|")}|<\\|s_(\\d+)\\|>`, "g");
  }
  encode(text) {
    const normalized = text.normalize("NFC");
    const ids = [];
    const re = this.addedSplitter;
    re.lastIndex = 0;
    let plainFrom = 0;
    for (let m = re.exec(normalized); m !== null; m = re.exec(normalized)) {
      let id;
      if (m[1] !== void 0) {
        const n = Number(m[1]);
        if (String(n) === m[1] && n < SPEECH_TOKEN_COUNT) id = SPEECH_TOKEN_BASE + n;
      } else {
        id = this.addedByContent.get(m[0]);
      }
      if (id === void 0) {
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
  encodePlain(span, ids) {
    if (span.length === 0) return;
    for (const [word] of span.matchAll(PRE_TOKENIZE)) {
      const cached = this.bpeCache.get(word);
      if (cached) {
        ids.push(...cached);
        continue;
      }
      const bytes = new TextEncoder().encode(word);
      let parts = new Array(bytes.length);
      for (let i = 0; i < bytes.length; i += 1) parts[i] = BYTE_TO_CHAR[bytes[i]];
      parts = this.merge(parts);
      const wordIds = parts.map((piece) => {
        const id = this.vocab.get(piece);
        if (id === void 0) {
          throw new Error(`piece ${JSON.stringify(piece)} not in vocab`);
        }
        return id;
      });
      this.bpeCache.set(word, wordIds);
      ids.push(...wordIds);
    }
  }
  /** Standard BPE: repeatedly apply the lowest-ranked adjacent merge. */
  merge(parts) {
    while (parts.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i < parts.length - 1; i += 1) {
        const rank = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== void 0 && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      const merged = parts[bestAt] + parts[bestAt + 1];
      const next = [];
      for (let i = 0; i < parts.length; i += 1) {
        if (i < parts.length - 1 && parts[i] === parts[bestAt] && parts[i + 1] === parts[bestAt + 1]) {
          next.push(merged);
          i += 1;
        } else {
          next.push(parts[i]);
        }
      }
      parts = next;
    }
    return parts;
  }
  decode(ids) {
    const decoder = new TextDecoder("utf-8", { fatal: false });
    let out = "";
    let pending = [];
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
      if (added !== void 0) {
        flush();
        out += added;
        continue;
      }
      const piece = this.pieces.get(id);
      if (piece === void 0) throw new Error(`id ${id} not in vocab`);
      for (const ch of piece) {
        const byte = CHAR_TO_BYTE.get(ch);
        if (byte === void 0) throw new Error(`piece char ${JSON.stringify(ch)} outside byte map`);
        pending.push(byte);
      }
    }
    flush();
    return out;
  }
  encodeChat(userText, opts) {
    const addGenerationPrompt = opts?.addGenerationPrompt ?? true;
    let prompt = "";
    if (opts?.system !== void 0) {
      prompt += `<|im_start|>system
${opts.system}<|im_end|>
`;
    }
    prompt += `<|im_start|>user
${userText}<|im_end|>
`;
    if (addGenerationPrompt) prompt += "<|im_start|>assistant\n";
    return this.encode(prompt);
  }
};
function splitMergeString(merge) {
  const at = merge.indexOf(" ");
  if (at < 0 || merge.indexOf(" ", at + 1) >= 0) {
    throw new Error(`unsplittable merge line ${JSON.stringify(merge)}`);
  }
  return [merge.slice(0, at), merge.slice(at + 1)];
}
async function loadTokenizer(json) {
  return new BpeTokenizer(json);
}

// browser.ts
var EOS_IDS = [151645, 151643];
var SAMPLE_RATE_EXPECTED = 24e3;
async function fetchWithProgress(url, label, report) {
  report(label);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
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
    report(label, { loaded, total });
  }
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}
function xorshift32(seed) {
  let s = seed >>> 0 || 2654435769;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}
async function requestDevice() {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu) throw new Error("WebGPU is unavailable \u2014 this page needs navigator.gpu for the language model");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("requestAdapter() returned null \u2014 no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize
    }
  });
  const info = adapter.info ? [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" ") || "unknown adapter" : "unknown adapter";
  return { device, adapter: info };
}
async function loadEverything(report) {
  const { device, adapter } = await requestDevice();
  report("loading tokenizer.json");
  const tokenizerJson = await (await fetch("./miotts/tokenizer.json")).json();
  const tokenizer = await loadTokenizer(tokenizerJson);
  report("loading the q8 manifest");
  const manifestResponse = await fetch("./miotts/q8/manifest.json");
  if (!manifestResponse.ok) throw new Error(`q8 manifest: HTTP ${manifestResponse.status}`);
  const manifest = await manifestResponse.json();
  const parts = [{ loaded: 0 }, { loaded: 0 }, { loaded: 0 }];
  const partReport = (index) => (_stage, detail) => {
    if (detail?.loaded === void 0) return;
    parts[index] = { loaded: detail.loaded, total: detail.total };
    const loaded = parts.reduce((sum, p) => sum + p.loaded, 0);
    const total = parts.every((p) => p.total !== void 0) ? parts.reduce((sum, p) => sum + (p.total ?? 0), 0) : void 0;
    report("LM weights (q8)", { loaded, total });
  };
  const [codes, scales, norms] = await Promise.all([
    fetchWithProgress(`./miotts/q8/${manifest.files.codes.name}`, "LM weights (q8)", partReport(0)),
    fetchWithProgress(`./miotts/q8/${manifest.files.scales.name}`, "LM weights (q8)", partReport(1)),
    fetchWithProgress(`./miotts/q8/${manifest.files.norms.name}`, "LM weights (q8)", partReport(2))
  ]);
  let checkpointVisible = false;
  const checkpointPromise = fetchCheckpoint((stage, detail) => {
    if (checkpointVisible) report(stage, detail);
  });
  checkpointPromise.catch(() => {
  });
  report("unpacking the q8 weights");
  await new Promise((resolve2) => setTimeout(resolve2, 0));
  const weights = loadWeightsQ8({ manifest, codes, scales, norms });
  report("uploading the LM to the GPU");
  const engine = await createGpuEngine(device, weights, { maxSeqLen: MAX_SEQ_LEN });
  checkpointVisible = true;
  report("downloading the checkpoint");
  const checkpoint = await checkpointPromise;
  report("parsing the codec checkpoint");
  const codecWeights = new Weights(Safetensors.parse(checkpoint));
  report("loading the speaker fixture");
  const fixtureResponse = await fetch("./mio-codec-fixture.json");
  if (!fixtureResponse.ok) throw new Error(`mio-codec-fixture.json: HTTP ${fixtureResponse.status}`);
  const fixture = await fixtureResponse.json();
  if (fixture.sample_rate !== SAMPLE_RATE_EXPECTED) {
    throw new Error(`fixture sample_rate ${fixture.sample_rate}, expected ${SAMPLE_RATE_EXPECTED}`);
  }
  const codecBackend = gpuBackend(Gpu.fromDevice(device, adapter)) ?? cpuBackend;
  return { adapter, engine, tokenizer, normalize: normalizeText, codecWeights, codecBackend, fixture };
}
async function generate(loaded, text, mode, seed, report) {
  const { engine, tokenizer, normalize, codecWeights, codecBackend, fixture } = loaded;
  const promptIds = tokenizer.encodeChat(normalize(text));
  if (promptIds.length + 1 >= MAX_SEQ_LEN) {
    throw new Error(`prompt is ${promptIds.length} tokens; the engine was built for maxSeqLen=${MAX_SEQ_LEN}`);
  }
  const maxNew = maxNewFor(promptIds.length);
  const sampler = mode === "greedy" ? { mode: "greedy" } : { mode: "top-p", temperature: 0.8, topP: 1, rng: xorshift32(seed) };
  engine.reset();
  report("generating speech tokens");
  const lmStart = performance.now();
  let lmSteps = 0;
  let logits = null;
  for (let i = 0; i < promptIds.length; i += 1) {
    logits = await engine.decodeStep(promptIds[i], { skipLogits: i + 1 < promptIds.length });
    lmSteps += 1;
  }
  const prefillMs = performance.now() - lmStart;
  const generatedIds = [];
  for (let step = 0; step < maxNew; step += 1) {
    const id = sampleNext(logits, generatedIds, sampler);
    generatedIds.push(id);
    if (EOS_IDS.includes(id)) break;
    if (step + 1 < maxNew) {
      logits = await engine.decodeStep(id);
      lmSteps += 1;
    }
    if (step % 10 === 0) report(`generating speech tokens \u2014 ${generatedIds.length}`);
  }
  const lmMs = performance.now() - lmStart;
  const speechIndices = [];
  const nonSpeechIds = [];
  for (const id of generatedIds) {
    if (EOS_IDS.includes(id)) continue;
    const index = speechIndexOf(id);
    if (index === null) nonSpeechIds.push(id);
    else speechIndices.push(index);
  }
  if (speechIndices.length === 0) throw new Error("the LM produced no speech tokens");
  report(`decoding ${speechIndices.length} speech tokens on ${codecBackend.name}`);
  await new Promise((resolve2) => setTimeout(resolve2, 0));
  const decodeStart = performance.now();
  const { waveform } = await decode(
    Float32Array.from(speechIndices),
    Float32Array.from(fixture.global_embedding),
    // The "aligned" path: 2 STFT frames per 25 Hz token -> n*960 samples at 24 kHz.
    2 * speechIndices.length,
    MIOCODEC_24K,
    codecWeights,
    codecBackend
  );
  const decodeMs = performance.now() - decodeStart;
  const audioSeconds = waveform.length / fixture.sample_rate;
  const totalMs = lmMs + decodeMs;
  return {
    result: {
      status: "done",
      adapter: loaded.adapter,
      mode,
      seed,
      promptIds,
      generatedIds,
      speechIndices,
      nonSpeechIds,
      prefillMs,
      lmMs,
      lmSteps,
      lmTokensPerSec: lmSteps / (lmMs / 1e3),
      decodeMs,
      totalMs,
      audioSeconds,
      rtf: totalMs / 1e3 / audioSeconds,
      engineStats: { ...engine.stats }
    },
    pcm: waveform,
    sampleRate: fixture.sample_rate
  };
}
function element(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error(`#${id} is missing from the page`);
  return node;
}
function main() {
  const button = element("run");
  const textArea = element("text");
  const modeSelect = element("mode");
  const seedInput = element("seed");
  const bar = element("bar");
  const status = element("status");
  const player = element("player");
  const metrics = element("metrics");
  window.__result = null;
  let loadedPromise = null;
  const formatMB = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  const report = (stage, detail) => {
    if (detail?.loaded !== void 0 && detail.total) {
      bar.value = detail.loaded / detail.total;
      status.textContent = `${stage} \u2014 ${formatMB(detail.loaded)} / ${formatMB(detail.total)}`;
    } else if (detail?.loaded !== void 0) {
      bar.removeAttribute("value");
      status.textContent = `${stage} \u2014 ${formatMB(detail.loaded)}`;
    } else {
      bar.removeAttribute("value");
      status.textContent = stage;
    }
  };
  button.addEventListener("click", async () => {
    button.disabled = true;
    bar.classList.remove("hidden");
    metrics.classList.add("hidden");
    window.__result = null;
    try {
      loadedPromise ??= loadEverything(report).catch((error) => {
        loadedPromise = null;
        throw error;
      });
      const loaded = await loadedPromise;
      const mode = modeSelect.value === "sample" ? "sample" : "greedy";
      const seedText = seedInput.value.trim();
      const parsedSeed = Number(seedText);
      const seed = seedText !== "" && Number.isFinite(parsedSeed) ? parsedSeed : 42;
      const { result, pcm, sampleRate } = await generate(loaded, textArea.value, mode, seed, report);
      bar.classList.add("hidden");
      status.textContent = "\u5B8C\u4E86";
      player.src = URL.createObjectURL(toWav(pcm, sampleRate));
      player.classList.remove("hidden");
      element("m-adapter").textContent = result.adapter;
      element("m-prompt-tokens").textContent = String(result.promptIds.length);
      element("m-gen-tokens").textContent = `${result.generatedIds.length} (${result.speechIndices.length} speech)` + (result.nonSpeechIds.length > 0 ? ` + ${result.nonSpeechIds.length} unexpected` : "");
      element("m-lm-ms").textContent = `${result.lmMs.toFixed(0)} ms (${result.lmSteps} steps)`;
      element("m-lm-tps").textContent = `${result.lmTokensPerSec.toFixed(1)} tok/s`;
      element("m-decode-ms").textContent = `${result.decodeMs.toFixed(0)} ms`;
      element("m-total-ms").textContent = `${result.totalMs.toFixed(0)} ms`;
      element("m-rtf").textContent = `${result.rtf.toFixed(2)} (${result.audioSeconds.toFixed(2)} s audio in ${(result.totalMs / 1e3).toFixed(2)} s)`;
      window.__result = result;
      metrics.classList.remove("hidden");
    } catch (error) {
      bar.classList.add("hidden");
      const message = error instanceof Error ? error.message : String(error);
      status.textContent = `\u5931\u6557: ${message}`;
      window.__result = { status: "error", error: message };
    } finally {
      button.disabled = false;
    }
  });
}
main();
