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
  /**
   * Styles the operator may select, out of those the generator offers.
   *
   * The generator's style is one piece of shared state -- whoever set it last set it for whoever
   * connects next -- so a scene that admits only one is not merely hiding the others: it asserts
   * its own on connect, rather than inheriting whatever the previous session was left on.
   */
  styles?: string[];
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
  /** Locomotion styles the generator offers, in selection order. */
  styles?: string[];
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

/**
 * Where the generator is, in order of preference: the page's own URL, then a `stream.json` beside
 * the page, then whatever the build declared.
 *
 * The middle one is what lets a single published link be steerable. A generator moves -- a tunnel
 * is restarted, a machine is replaced -- and if its address is compiled into the bundle then every
 * move costs a rebuild and a redeploy of an eighty-megabyte page. As a file next to the page it is
 * a hundred bytes, and the link never changes.
 *
 * Absent or unreachable, the page stays exactly what it is without a generator: a self-contained
 * demo of a recorded clip. That is the honest default for a public link, since the machine at the
 * far end will not always be up.
 */
export async function resolveStreamConfig(
  declared?: LiveMotionStreamConfig,
): Promise<LiveMotionStreamConfig | null> {
  const search = typeof window === 'undefined' ? '' : window.location?.search ?? '';
  const override = new URLSearchParams(search).get('stream');
  if (override) {
    return { ...(declared ?? {}), url: override };
  }
  if (typeof document !== 'undefined') {
    try {
      // `no-store`: the whole point is that this can change between visits.
      const response = await fetch(new URL('stream.json', document.baseURI), { cache: 'no-store' });
      if (response.ok) {
        const published = (await response.json()) as Partial<LiveMotionStreamConfig>;
        if (published?.url) {
          return { ...(declared ?? {}), ...published, url: published.url };
        }
      }
    } catch {
      // A missing or unreachable stream.json is the normal case for a clip-only deploy.
    }
  }
  return declared ?? null;
}

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
  private styles: string[] = [];
  private styleIndex = 0;
  private stylePanel: HTMLElement | null = null;

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
          const offered = message.styles ?? [];
          const allowed = this.config.styles;
          // The generator's order is kept: the allow-list says which, not in what order.
          this.styles = allowed ? offered.filter((name) => allowed.includes(name)) : offered;
          if (this.styles.length === 1) {
            socket.send(JSON.stringify({ type: 'style', name: this.styles[0] }));
          }
          this.renderStyles();
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

  /**
   * Choose a locomotion style by its position in the list the generator announced.
   *
   * Number keys rather than letters: the letters a keyboard-driven robot can spare are already
   * taken by the direction and turn keys, and the styles are a list whose contents come from the
   * far end of the socket rather than a fixed set worth memorising.
   */
  selectStyle(index: number): void {
    if (index < 0 || index >= this.styles.length || index === this.styleIndex) {
      return;
    }
    this.styleIndex = index;
    if (this.connected) {
      this.socket?.send(JSON.stringify({ type: 'style', name: this.styles[index] }));
    }
    this.renderStyles();
  }

  /** A small panel listing the styles, so the number keys are discoverable. */
  private renderStyles(): void {
    // Nothing to choose between is nothing to draw, and the number keys are inert anyway.
    if (typeof document === 'undefined' || this.styles.length < 2) {
      return;
    }
    if (!this.stylePanel) {
      const panel = document.createElement('div');
      panel.style.cssText = [
        'position:fixed', 'left:12px', 'bottom:12px', 'z-index:40',
        'font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace',
        'background:rgba(17,20,24,0.82)', 'color:#c8ced6',
        'border:1px solid rgba(255,255,255,0.10)', 'border-radius:8px',
        'padding:8px 10px', 'pointer-events:none', 'backdrop-filter:blur(6px)',
      ].join(';');
      document.body.appendChild(panel);
      this.stylePanel = panel;
    }
    const rows = this.styles
      .map((name, i) => {
        const label = name.replace(/_/g, ' ');
        const active = i === this.styleIndex;
        const colour = active ? '#8fd694' : '#c8ced6';
        const marker = active ? '&#9679;' : '&nbsp;';
        return `<div style="color:${colour}">${marker} ${i + 1}&nbsp; ${label}</div>`;
      })
      .join('');
    this.stylePanel.innerHTML =
      `<div style="color:#8a93a0;margin-bottom:4px">style</div>${rows}`;
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
      if (key >= '1' && key <= '9') {
        this.selectStyle(Number(key) - 1);
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
    this.stylePanel?.remove();
    this.stylePanel = null;
    this.detachKeys?.();
    this.detachKeys = null;
    this.socket?.close();
    this.socket = null;
  }
}
