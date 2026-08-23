import { conv1d } from "web-xpu-ops/ops/conv";
import { convTranspose1d } from "web-xpu-ops/ops/conv_transpose";
import { snake } from "web-xpu-ops/ops/snake";

/**
 * The three operations the decode path is made of, behind one seam.
 *
 * Same shape as `spike/miocodec`'s `Backend`, and for the same reason: the CPU
 * and GPU paths should be the same graph with one object swapped, so a
 * disagreement between them is a disagreement about arithmetic and not about
 * which stages ran.
 *
 * Unlike MioCodec's, this covers the *whole* graph rather than the three
 * expensive ops out of a dozen. DACVAE's decoder is only these three, so there
 * is no hybrid here and nothing quietly running on the CPU in the GPU path.
 */
export interface Backend {
  readonly name: string;
  conv1d(args: {
    input: Float32Array;
    weight: Float32Array;
    bias: Float32Array | undefined;
    Cin: number;
    Cout: number;
    L: number;
    K: number;
    padding: number;
    dilation: number;
  }): Promise<Float32Array>;
  convTranspose1d(args: {
    input: Float32Array;
    weight: Float32Array;
    bias: Float32Array | undefined;
    Cin: number;
    Cout: number;
    L: number;
    K: number;
    stride: number;
    padding: number;
    outputPadding: number;
  }): Promise<Float32Array>;
  snake(args: {
    input: Float32Array;
    alpha: Float32Array;
    C: number;
    L: number;
  }): Promise<Float32Array>;
}

/** The reference implementations — the definition of correct, and the slowest. */
export const cpuBackend: Backend = {
  name: "reference (CPU)",
  async conv1d({ input, weight, bias, Cin, Cout, L, K, padding, dilation }) {
    return conv1d({ input, weight, ...(bias ? { bias } : {}), N: 1, Cin, Cout, L, K, padding, dilation });
  },
  async convTranspose1d({ input, weight, bias, Cin, Cout, L, K, stride, padding, outputPadding }) {
    return convTranspose1d({
      input,
      weight,
      ...(bias ? { bias } : {}),
      N: 1,
      Cin,
      Cout,
      L,
      K,
      stride,
      padding,
      outputPadding,
    });
  },
  async snake({ input, alpha, C, L }) {
    return snake({ input, alpha, N: 1, C, L });
  },
};
