import * as ort from 'onnxruntime-web';

import { queueOrtRun } from '../onnx/runQueue';

// Main thread, not ORT's proxy worker: the bundled worker entry touches `document` and throws
// `ReferenceError: document is not defined` on start. The cost is that a graph heavier than the
// frame budget — a traced whole-body IK, say — blocks rendering for its duration rather than
// merely missing a control step. Moving inference off-thread needs the worker chunk emitted
// properly by the bundler first.
ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = 1;
// Lib build: redirect ort's dynamic file fetches to its own CDN package.
// __ORT_CDN_BASE__ is defined only by vite.lib.config.ts (no-op in the SPA).
declare const __ORT_CDN_BASE__: string | undefined;
if (typeof __ORT_CDN_BASE__ !== 'undefined') {
  ort.env.wasm.wasmPaths = __ORT_CDN_BASE__;
}

export type OnnxMeta = {
  in_keys?: string[];
  out_keys?: (string | string[])[];
};

export class OnnxModule {
  private bytes: ArrayBuffer;
  private session: ort.InferenceSession | null;
  private configuredInKeys: string[];
  inKeys: string[];
  outKeys: string[];
  isRecurrent: boolean;

  constructor(bytes: ArrayBuffer, meta?: OnnxMeta) {
    this.bytes = bytes;
    this.session = null;
    const inKeys = meta?.in_keys ?? ['policy'];
    const outKeys = meta?.out_keys ?? ['action'];
    this.configuredInKeys = inKeys.map((key) => (Array.isArray(key) ? key.join(',') : key));
    this.inKeys = [...this.configuredInKeys];
    this.outKeys = outKeys.map((key) => (Array.isArray(key) ? key.join(',') : key));
    this.isRecurrent = this.inKeys.includes('adapt_hx');
  }

  async init(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    this.inferInputKeys();
    this.isRecurrent = this.inKeys.includes('adapt_hx');
  }

  initInput(): Record<string, ort.Tensor> {
    const inputs: Record<string, ort.Tensor> = {};
    if (this.isRecurrent) {
      inputs.is_init = new ort.Tensor('bool', [true], [1]);
      inputs.adapt_hx = new ort.Tensor('float32', new Float32Array(128), [1, 128]);
    }
    if (this.inKeys.includes('time_step')) {
      inputs.time_step = new ort.Tensor('float32', new Float32Array([0.0]), [1, 1]);
    }
    return inputs;
  }

  async runInference(
    input: Record<string, ort.Tensor>
  ): Promise<[Record<string, ort.Tensor>, Record<string, ort.Tensor>]> {
    if (!this.session) {
      throw new Error('OnnxModule not initialized.');
    }

    const onnxInput: Record<string, ort.Tensor> = {};
    for (let i = 0; i < this.inKeys.length; i++) {
      const key = this.inKeys[i];
      const name = this.session.inputNames[i];
      if (!name || !input[key]) {
        throw new Error(`Missing ONNX input for key: ${key}`);
      }
      onnxInput[name] = input[key];
    }

    const session = this.session;
    const onnxOutput = await queueOrtRun(() => session.run(onnxInput));
    const result: Record<string, ort.Tensor> = {};
    for (let i = 0; i < this.outKeys.length; i++) {
      const key = this.outKeys[i];
      const name = this.session.outputNames[i];
      if (name && onnxOutput[name]) {
        result[key] = onnxOutput[name];
      }
    }

    const carry: Record<string, ort.Tensor> = {};
    if (this.isRecurrent && result['next,adapt_hx']) {
      carry.is_init = new ort.Tensor('bool', [false], [1]);
      carry.adapt_hx = result['next,adapt_hx'];
    }

    return [result, carry];
  }

  /** Release the ONNX Runtime session, freeing its WASM memory. */
  dispose(): void {
    void this.session?.release?.();
    this.session = null;
  }

  private inferInputKeys(): void {
    if (!this.session) {
      return;
    }
    const modelInputs = this.session.inputNames;
    if (modelInputs.length <= 1) {
      this.inKeys = [...this.configuredInKeys];
      return;
    }

    if (this.configuredInKeys.length === modelInputs.length) {
      this.inKeys = [...this.configuredInKeys];
      return;
    }

    this.inKeys = modelInputs.map((name) => {
      if (this.configuredInKeys.includes(name)) {
        return name;
      }
      if (name === 'obs' && this.configuredInKeys.includes('policy')) {
        return 'policy';
      }
      // No configured key matches: `runInference` forwards it under the model's own name.
      return name;
    });
  }
}
