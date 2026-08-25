/**
 * A reference motion that arrives over a websocket while the episode is running.
 *
 * A bundled clip is a fixed array of frames; this is the same thing fed by a generator that has not
 * finished yet. The engine needs no other change: frames land in the same per-frame `Float32Array`
 * layout `TrackingCommand` already indexes, and the arrays simply grow.
 *
 * The motivating case is steering a tracker from the keyboard. A tracking policy follows a
 * reference, so a velocity command can only reach the robot by way of *something that invents the
 * reference* -- here a motion model on a GPU somewhere else, since it does not fit in a tab. The
 * command goes up, frames come down.
 *
 * Two consequences worth knowing before using this:
 *
 * - **The frames are authoritative, not advisory.** Nothing re-anchors them to the robot's measured
 *   pose, so the generator has to produce a continuous world-frame trajectory. That is what lets
 *   the buffer be appended to blindly.
 * - **Buffer depth is steering latency.** Everything already fetched was generated under an older
 *   command, so a deep buffer makes the keyboard feel late. `lead` has to cover the policy's
 *   look-ahead window and the round trip, and not much more.
 */

export type LiveMotionStreamConfig = {
  /** Websocket URL of the reference server. */
  url: string;
  /** Frames to keep fetched beyond what is being read. See the note on latency above. */
  lead?: number;
  /** Frames per request. Larger amortizes the round trip; smaller spreads the work out. */
  block?: number;
  /** Attach WASD key handling to the window. */
  keys?: boolean;
};

/** The per-frame arrays a tracking clip is made of, in the engine's own layout. */
export type LiveFrameArrays = {
  jointPos: Float32Array[];
  jointVel: Float32Array[];
  bodyPosW: Float32Array[];
  bodyQuatW: Float32Array[];
  bodyLinVelW: Float32Array[];
  bodyAngVelW: Float32Array[];
};

type Hello = {
  type: string;
  control_dt: number;
  n_dofs: number;
  body_names: string[];
};

// Buffer depth is not free: it is steering latency, and it is also dead time in the loop that
// keeps the reference within reach of the robot, so a deep buffer makes the correction act on
// stale information (measured: buffering ~2 s left the robot 2 m behind, against 0.45 m at ~1 s).
// The floor is the policy's own look-ahead window plus a round trip.
const DEFAULT_LEAD = 40;
const DEFAULT_BLOCK = 25;

/**
 * What each key contributes to the command, as (forward m/s, lateral m/s, turn deg/s) in the
 * robot's own frame. Contributions add, so W+A walks forward and left rather than one or the other.
 */
const KEY_COMMANDS: Record<string, [number, number, number]> = {
  w: [0.8, 0.0, 0.0],
  s: [-0.5, 0.0, 0.0],
  a: [0.0, 0.45, 0.0],
  d: [0.0, -0.45, 0.0],
  q: [0.4, 0.0, 20.0],
  e: [0.4, 0.0, -20.0],
  arrowup: [0.8, 0.0, 0.0],
  arrowdown: [-0.5, 0.0, 0.0],
  arrowleft: [0.4, 0.0, 20.0],
  arrowright: [0.4, 0.0, -20.0],
};

function emptyFrames(): LiveFrameArrays {
  return {
    jointPos: [],
    jointVel: [],
    bodyPosW: [],
    bodyQuatW: [],
    bodyLinVelW: [],
    bodyAngVelW: [],
  };
}

/**
 * Split one binary frame block into per-frame arrays.
 *
 * Layout, matching the server: `int32 start | int32 count | float32 payload`, the payload being
 * each field's frames in turn -- joint positions and velocities, then body positions, orientations,
 * linear and angular velocities. Binary rather than JSON because a frame is ~487 floats and this
 * runs at the control rate.
 */
export function parseFrameBlock(
  buffer: ArrayBuffer,
  nDofs: number,
  nBodies: number,
): { start: number; count: number; frames: LiveFrameArrays } {
  const header = new DataView(buffer);
  const start = header.getInt32(0, true);
  const count = header.getInt32(4, true);
  const frames = emptyFrames();
  let offset = 8;

  const take = (target: Float32Array[], width: number): void => {
    for (let i = 0; i < count; i++) {
      // Copied rather than sub-arrayed: a view would pin the whole received buffer for as long as
      // any frame in it is still in the ring.
      target.push(new Float32Array(buffer, offset + i * width * 4, width).slice());
    }
    offset += count * width * 4;
  };

  take(frames.jointPos, nDofs);
  take(frames.jointVel, nDofs);
  take(frames.bodyPosW, nBodies * 3);
  take(frames.bodyQuatW, nBodies * 4);
  take(frames.bodyLinVelW, nBodies * 3);
  take(frames.bodyAngVelW, nBodies * 3);
  return { start, count, frames };
}

export class LiveMotionSource {
  readonly frames: LiveFrameArrays = emptyFrames();
  /** Resolves once the server's hello has been read, so dimensions are known. */
  readonly ready: Promise<void>;
  private socket: WebSocket | null = null;
  private hello: Hello | null = null;
  private readonly lead: number;
  private readonly block: number;
  /** Next frame index to ask for; requests never overlap, so blocks abut and can be appended. */
  private requestedTo = 0;
  private command: [number, number, number] = [0, 0, 0];
  private readonly pressed = new Set<string>();
  private resolveReady: (() => void) | null = null;
  private detachKeys: (() => void) | null = null;

  constructor(private readonly config: LiveMotionStreamConfig) {
    this.lead = config.lead ?? DEFAULT_LEAD;
    this.block = config.block ?? DEFAULT_BLOCK;
    this.ready = new Promise((resolve) => {
      this.resolveReady = resolve;
    });
  }

  get length(): number {
    return this.frames.jointPos.length;
  }

  get controlDt(): number {
    return this.hello?.control_dt ?? 0.02;
  }

  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  connect(): void {
    const socket = new WebSocket(this.config.url);
    socket.binaryType = 'arraybuffer';
    this.socket = socket;

    socket.onmessage = (event: MessageEvent): void => {
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data) as Hello;
        if (message.type === 'hello') {
          this.hello = message;
          this.resolveReady?.();
          this.resolveReady = null;
          // Nothing is buffered yet, so the first request has to be issued here rather than
          // waiting for a read: the engine cannot advance a clip with no frames in it.
          this.request();
        }
        return;
      }
      this.append(event.data as ArrayBuffer);
    };
    socket.onerror = (): void => {
      console.error(`[liveMotion] websocket error on ${this.config.url}`);
    };
    socket.onclose = (): void => {
      console.warn('[liveMotion] reference stream closed; the clip stops growing');
    };

    if (this.config.keys !== false) {
      this.attachKeyboard();
    }
  }

  private append(buffer: ArrayBuffer): void {
    if (!this.hello) {
      return;
    }
    const { start, count, frames } = parseFrameBlock(
      buffer,
      this.hello.n_dofs,
      this.hello.body_names.length,
    );
    if (start !== this.length) {
      // Out-of-order or duplicated blocks would tear the trajectory; refusing is better than
      // splicing a gap the tracker would read straight through.
      console.error(`[liveMotion] expected frame ${this.length}, got ${start}; dropping block`);
      return;
    }
    for (const key of Object.keys(frames) as (keyof LiveFrameArrays)[]) {
      for (const frame of frames[key]) {
        this.frames[key].push(frame);
      }
    }
    void count;
  }

  private request(): void {
    if (!this.connected) {
      return;
    }
    this.socket?.send(
      JSON.stringify({ type: 'request', from: this.requestedTo, count: this.block }),
    );
    this.requestedTo += this.block;
  }

  /** Keep the buffer filled past ``index``; cheap and idempotent, safe to call every step. */
  ensure(index: number): void {
    while (this.requestedTo <= index + this.lead) {
      this.request();
      if (!this.connected) {
        return;
      }
    }
  }

  /**
   * Report the robot's own pose, as `qpos` in the contract's order.
   *
   * The generator cannot see this simulation, so this is the only thing keeping it honest: a
   * kinematic reference is not something a physical gait matches exactly, and the shortfall
   * accumulates until the operator is steering a reference the robot is nowhere near. A generator
   * that continues from a pose window uses this to steer where it generates from. ``frame`` is the
   * reference frame the robot is currently on, without which the generator would read its own
   * look-ahead buffer as tracking error and over-correct by the buffer depth.
   */
  reportContext(qpos: ArrayLike<number>, frame: number, refQuat?: ArrayLike<number>): void {
    if (this.connected) {
      this.socket?.send(
        JSON.stringify({
          type: 'context',
          qpos: Array.from(qpos),
          frame,
          // The reference orientation this client is actually reading, so the generator can check
          // that what arrived over the wire is what it sent -- an indexing or byte-order fault
          // otherwise looks exactly like a policy that ignores the reference.
          ref_quat: refQuat ? Array.from(refQuat) : undefined,
        }),
      );
    }
  }

  setCommand(forward: number, lateral: number, turn: number): void {
    if (
      forward === this.command[0] &&
      lateral === this.command[1] &&
      turn === this.command[2]
    ) {
      return; // Only changes are worth a message; keys repeat while held.
    }
    this.command = [forward, lateral, turn];
    if (this.connected) {
      this.socket?.send(JSON.stringify({ type: 'command', forward, lateral, turn }));
    }
  }

  getCommand(): [number, number, number] {
    return [...this.command];
  }

  private recomputeCommand(): void {
    let forward = 0;
    let lateral = 0;
    let turn = 0;
    for (const key of this.pressed) {
      const contribution = KEY_COMMANDS[key];
      if (contribution) {
        forward += contribution[0];
        lateral += contribution[1];
        turn += contribution[2];
      }
    }
    this.setCommand(forward, lateral, turn);
  }

  attachKeyboard(target: Window | null = typeof window === 'undefined' ? null : window): void {
    if (!target || this.detachKeys) {
      return;
    }
    const down = (event: KeyboardEvent): void => {
      const key = event.key.toLowerCase();
      if (key === ' ') {
        this.pressed.clear();
        this.recomputeCommand();
        return;
      }
      if (!(key in KEY_COMMANDS) || event.repeat) {
        return;
      }
      this.pressed.add(key);
      this.recomputeCommand();
    };
    const up = (event: KeyboardEvent): void => {
      const key = event.key.toLowerCase();
      if (this.pressed.delete(key)) {
        this.recomputeCommand();
      }
    };
    // Releasing outside the page would otherwise leave the robot walking with no key held.
    const blur = (): void => {
      this.pressed.clear();
      this.recomputeCommand();
    };
    target.addEventListener('keydown', down);
    target.addEventListener('keyup', up);
    target.addEventListener('blur', blur);
    this.detachKeys = (): void => {
      target.removeEventListener('keydown', down);
      target.removeEventListener('keyup', up);
      target.removeEventListener('blur', blur);
    };
  }

  dispose(): void {
    this.detachKeys?.();
    this.detachKeys = null;
    this.socket?.close();
    this.socket = null;
  }
}
