/**
 * The virtual contact a force-exertion policy pushes against, and the gauge that reads it.
 *
 * A policy trained to *produce* a hand force is trained against a Kelvin-Voigt contact anchored at
 * the reference hand: the hand leads along a push axis, the contact resists, and the reaction loads
 * the whole body. Without that reaction the hand simply moves to its displaced target, nothing is
 * exerted, and there is nothing to measure — the arm lead is passive. This applies the reaction and
 * publishes the force it implies, which is the same quantity the training reward measured.
 *
 *     lead = n_robot . (x_hand - anchor)  -  n_ref . (x_ref - ref_anchor)
 *     p    = clamp(lead, p_max)                       (smooth, optionally two-sided)
 *     F    = K_v * p + C_v * dp/dt                    along the push axis
 *     robot receives -F * n_robot at the hand
 *
 * The lead is measured in each frame's *own* anchor — the robot's hand against the robot's torso,
 * the reference hand against the reference torso — so the force does not move with global tracking
 * drift. That is `local_frame_lead` in the training config, and it is why the robot-side push axis
 * has to be rotated by the robot's heading here rather than taken from the graph.
 *
 * Everything the reference side needs (`K_v`, `C_v`, the push axis, the reference hand and anchor)
 * comes from the brace graph's published state, so the browser never re-derives the training
 * constants.
 */

import * as THREE from 'three';

import { mjcToThreeCoordinate } from '../scene/coordinate';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** Metres of arrow per newton. 9 N (the trained ceiling) reads as a forearm. */
const ARROW_M_PER_N = 0.05;
const COMMANDED_COLOR = 0x3fa9f5;
const EXERTED_COLOR = 0xf5a03f;

export interface HandSpringTarget {
  /** Body the force acts on. */
  body: string;
  /** Its index in the policy's body order, for slicing the graph's per-hand arrays. */
  hand: number;
}

export interface HandSpringConfig {
  /** Command term publishing the brace state. */
  command_name: string;
  /** Body whose heading defines the robot-side frame (the policy's anchor). */
  anchor_body: string;
  targets: HandSpringTarget[];
  /** Max lead the contact supports, metres (`max_lead`). */
  max_lead: number;
  /** Softplus knee width for the lead clamp, 1/m (`smooth_contact_beta`); 0 disables smoothing. */
  smooth_beta?: number;
  /** Symmetric lead, so the contact pulls as well as pushes (`two_sided_spring`). */
  two_sided?: boolean;
  /** Include the `C_v * dp/dt` damper term (`contact_damping`). */
  damping?: boolean;
  /** Control step, for the lead's finite difference. */
  dt: number;
  /** Draw the commanded and exerted force at each hand. */
  gauge?: boolean;
}

/**
 * A canvas-backed sprite showing the gauge's two numbers.
 *
 * In the scene rather than the control panel: the panel has no read-only widget, and a reading
 * pinned to the hand it describes is easier to follow than one in a list — especially with two
 * hands. Redrawn only when the text changes, so the per-frame cost is a comparison.
 */
class ForceLabel {
  private readonly canvas = document.createElement('canvas');
  private readonly texture: THREE.CanvasTexture;
  readonly sprite: THREE.Sprite;
  private last = '';

  constructor() {
    this.canvas.width = 256;
    this.canvas.height = 112;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: this.texture, transparent: true, depthWrite: false }),
    );
    this.sprite.scale.set(0.40, 0.175, 1);
    this.sprite.visible = false;
  }

  set(commanded: number, exerted: number): void {
    const text = `cmd ${commanded.toFixed(1)} N   exert ${exerted.toFixed(1)} N`;
    if (text !== this.last) {
      const ctx = this.canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = 'rgba(12, 16, 20, 0.72)';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        // Two lines: at a glance the pair is a comparison, and one line ran them together.
        ctx.font = 'bold 30px system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#3fa9f5';
        ctx.fillText(`cmd   ${commanded.toFixed(1)} N`, 12, 32);
        ctx.fillStyle = '#f5a03f';
        ctx.fillText(`exert ${exerted.toFixed(1)} N`, 12, 78);
        this.texture.needsUpdate = true;
      }
      this.last = text;
    }
    this.sprite.visible = true;
  }

  hide(): void {
    this.sprite.visible = false;
  }

  dispose(): void {
    this.sprite.parent?.remove(this.sprite);
    this.texture.dispose();
    this.sprite.material.dispose();
  }
}

/** Shaft + head, unit length along +Y with its base at the origin. */
function makeArrow(color: number): THREE.Group {
  const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 1, 10).translate(0, 0.5, 0), material);
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.06, 10).translate(0, 0.97, 0), material);
  const group = new THREE.Group();
  group.add(shaft, head);
  group.visible = false;
  return group;
}

/** Place `arrow` from `origin` along `vec` (MuJoCo world), hiding it when the force is ~0. */
function placeArrow(arrow: THREE.Group, origin: ArrayLike<number>, vec: readonly number[]): void {
  const magnitude = Math.hypot(vec[0], vec[1], vec[2]);
  if (magnitude < 1e-3) {
    arrow.visible = false;
    return;
  }
  // Both ends converted, not the direction: the MuJoCo-to-three mapping is an axis swizzle, so
  // transforming a vector on its own and a point are not the same operation.
  const start = mjcToThreeCoordinate([origin[0], origin[1], origin[2]]);
  const tip = mjcToThreeCoordinate([origin[0] + vec[0], origin[1] + vec[1], origin[2] + vec[2]]);
  const dir = tip.clone().sub(start);
  const length = dir.length();
  arrow.position.copy(start);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  arrow.scale.set(1, length, 1);
  arrow.visible = true;
}

interface StateSource {
  getStateField?(field: string): Float32Array | null;
}

function decodeBodyNames(mjModel: MjModel): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let b = 0; b < mjModel.nbody; b++) {
    const start = mjModel.name_bodyadr[b];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

function resolveBody(names: string[], name: string): number {
  const direct = names.indexOf(name);
  if (direct > 0) return direct;
  return names.findIndex((n) => n.endsWith(`/${name}`));
}

/** Rotate `v` by the yaw-only part of a wxyz quaternion — the heading frame. */
function applyHeading(quatWxyz: ArrayLike<number>, v: readonly number[]): [number, number, number] {
  const [w, x, y, z] = [quatWxyz[0], quatWxyz[1], quatWxyz[2], quatWxyz[3]];
  // Yaw of the quaternion, then a plain z-rotation: the push axis is defined against heading, not
  // against the full orientation, so pitch and roll must not turn it.
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
}

export class HandSpringContact {
  private readonly bodyIds: number[];
  private readonly anchorId: number;
  /** Previous clamped lead per hand, for the damper's finite difference. */
  private readonly prevLead: Float32Array;
  private primed = false;
  /** Last measured along-axis force per hand, newtons — what the gauge reads. */
  readonly measured: Float32Array;
  private readonly commandedArrows: THREE.Group[] = [];
  private readonly exertedArrows: THREE.Group[] = [];
  private readonly labels: ForceLabel[] = [];

  constructor(
    private readonly config: HandSpringConfig,
    mjModel: MjModel,
    parent?: THREE.Object3D | null,
  ) {
    const names = decodeBodyNames(mjModel);
    this.bodyIds = config.targets.map((t) => resolveBody(names, t.body));
    this.anchorId = resolveBody(names, config.anchor_body);
    this.prevLead = new Float32Array(config.targets.length);
    this.measured = new Float32Array(config.targets.length);
    if (config.gauge && parent) {
      for (let i = 0; i < config.targets.length; i++) {
        const commanded = makeArrow(COMMANDED_COLOR);
        const exerted = makeArrow(EXERTED_COLOR);
        commanded.name = `hand-force-commanded-${i}`;
        exerted.name = `hand-force-exerted-${i}`;
        parent.add(commanded, exerted);
        this.commandedArrows.push(commanded);
        this.exertedArrows.push(exerted);
        const label = new ForceLabel();
        parent.add(label.sprite);
        this.labels.push(label);
      }
    }
  }

  /** One-sided or symmetric lead clamp, softened as `smooth_contact` does. */
  private clampLead(lead: number): number {
    const pMax = this.config.max_lead;
    if (this.config.two_sided) return pMax * Math.tanh(lead / Math.max(pMax, 1e-9));
    const beta = this.config.smooth_beta ?? 0;
    // Softplus knee at 0, then the same tanh saturation at p_max.
    const positive = beta > 0 ? Math.log1p(Math.exp(Math.min(beta * lead, 30))) / beta : Math.max(lead, 0);
    return pMax * Math.tanh(positive / Math.max(pMax, 1e-9));
  }

  /**
   * Apply this step's reaction and refresh {@link measured}.
   *
   * Writes (not accumulates) its target bodies, so a hand with no command reliably returns to zero.
   */
  apply(mjData: MjData, brace: StateSource | undefined): void {
    const read = (field: string) => brace?.getStateField?.(field) ?? null;
    const kv = read('endpoint_kv');
    const cv = read('endpoint_cv');
    const axisRef = read('push_axis_w');
    const axisLocal = read('push_axis_local');
    const refHand = read('ref_hand_pos');
    const refAnchor = read('ref_anchor_pos');
    if (!kv || !axisRef || !axisLocal || !refHand || !refAnchor) {
      this.measured.fill(0);
      for (const a of [...this.commandedArrows, ...this.exertedArrows]) a.visible = false;
      for (const l of this.labels) l.hide();
      return;
    }

    const anchor = [
      mjData.xpos[this.anchorId * 3],
      mjData.xpos[this.anchorId * 3 + 1],
      mjData.xpos[this.anchorId * 3 + 2],
    ];
    const anchorQuat = [
      mjData.xquat[this.anchorId * 4],
      mjData.xquat[this.anchorId * 4 + 1],
      mjData.xquat[this.anchorId * 4 + 2],
      mjData.xquat[this.anchorId * 4 + 3],
    ];

    for (let t = 0; t < this.config.targets.length; t++) {
      const bodyId = this.bodyIds[t];
      const h = this.config.targets[t].hand;
      const base = bodyId * 6;
      if (bodyId <= 0) continue;

      // The dial's direction carried into the robot's own heading.
      const nRobot = applyHeading(anchorQuat, [
        axisLocal[h * 3],
        axisLocal[h * 3 + 1],
        axisLocal[h * 3 + 2],
      ]);
      const active = Math.hypot(nRobot[0], nRobot[1], nRobot[2]);
      if (active < 1e-6) {
        for (let i = 0; i < 6; i++) mjData.xfrc_applied[base + i] = 0;
        this.measured[t] = 0;
        this.prevLead[t] = 0;
        if (this.commandedArrows[t]) {
          this.commandedArrows[t].visible = false;
          this.exertedArrows[t].visible = false;
          this.labels[t].hide();
        }
        continue;
      }

      let robotLead = 0;
      let refLead = 0;
      for (let i = 0; i < 3; i++) {
        robotLead += nRobot[i] * (mjData.xpos[bodyId * 3 + i] - anchor[i]);
        refLead += axisRef[h * 3 + i] * (refHand[h * 3 + i] - refAnchor[i]);
      }
      const p = this.clampLead(robotLead - refLead);
      // First step after a reset has no previous lead; a finite difference against zero would be a
      // spike straight into the damper.
      const pDot = this.primed ? (p - this.prevLead[t]) / this.config.dt : 0;
      this.prevLead[t] = p;

      let force = kv[h] * p;
      if (this.config.damping && cv) force += cv[h] * pDot;
      this.measured[t] = force;

      // Reaction on the robot: the contact pushes back along -n.
      for (let i = 0; i < 3; i++) mjData.xfrc_applied[base + i] = -force * nRobot[i];
      for (let i = 3; i < 6; i++) mjData.xfrc_applied[base + i] = 0;

      // The gauge: both arrows start at the hand, so their lengths compare directly. Commanded is
      // what was asked for after the feasibility caps; exerted is what the contact actually reads.
      const hand = [
        mjData.xpos[bodyId * 3],
        mjData.xpos[bodyId * 3 + 1],
        mjData.xpos[bodyId * 3 + 2],
      ];
      if (this.commandedArrows[t]) {
        const cmdW = read('force_cmd_w');
        const cmd = cmdW
          ? [cmdW[h * 3] * ARROW_M_PER_N, cmdW[h * 3 + 1] * ARROW_M_PER_N, cmdW[h * 3 + 2] * ARROW_M_PER_N]
          : [0, 0, 0];
        placeArrow(this.commandedArrows[t], hand, cmd);
        placeArrow(this.exertedArrows[t], hand, [
          force * nRobot[0] * ARROW_M_PER_N,
          force * nRobot[1] * ARROW_M_PER_N,
          force * nRobot[2] * ARROW_M_PER_N,
        ]);
        // The commanded magnitude is the post-cap one the policy observes, not the raw dial, so
        // the two numbers are directly comparable: what was asked for against what is being felt.
        const cmdMag = cmdW ? Math.hypot(cmdW[h * 3], cmdW[h * 3 + 1], cmdW[h * 3 + 2]) : 0;
        this.labels[t].set(cmdMag, force);
        const above = mjcToThreeCoordinate([hand[0], hand[1], hand[2] + 0.16]);
        this.labels[t].sprite.position.copy(above);
      }
    }
    this.primed = true;
  }

  reset(): void {
    this.prevLead.fill(0);
    this.measured.fill(0);
    this.primed = false;
  }
}
