/**
 * The MuJoCo side of hand tracking: binds the injected rig (handRig.ts) to the compiled
 * model, writes tracked joint poses into `mocap_pos`/`mocap_quat`, and turns a pinch into
 * an equality weld so a grabbed body is actually held rather than nudged.
 *
 * Everything here is in MuJoCo coordinates — the Three.js side converts before handing
 * poses over, so no swizzle leaks into the physics.
 */

import { quatApplyInv, quatInverse, quatMultiply, normalizeQuat } from '../observation/math';
import {
  HANDEDNESS,
  HAND_JOINTS,
  PALM_JOINT,
  PARK_POS,
  PINCH_JOINTS,
  type Handedness,
  handBodyName,
  handWeldName,
} from './handRig';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** A joint pose in MuJoCo coordinates: position, then (w, x, y, z) orientation. */
export interface HandJointPose {
  pos: [number, number, number];
  quat: [number, number, number, number];
}

export interface HandFrame {
  /** The XR runtime reported this hand on the last snapshot. */
  tracked: boolean;
  pinching: boolean;
  /** One slot per `HAND_JOINTS`, null where the runtime reported no pose. */
  joints: (HandJointPose | null)[];
}

export type HandFrames = Record<Handedness, HandFrame>;

export function createHandFrames(): HandFrames {
  const frame = (): HandFrame => ({
    tracked: false,
    pinching: false,
    joints: HAND_JOINTS.map(() => null),
  });
  return { left: frame(), right: frame() };
}

/** Surface distance from the pinch point within which a body can be picked up. */
const GRAB_REACH = 0.05;

const PALM_INDEX = HAND_JOINTS.findIndex((entry) => entry.joint === PALM_JOINT);
const PINCH_INDICES = PINCH_JOINTS.map((joint) =>
  HAND_JOINTS.findIndex((entry) => entry.joint === joint)
);

interface JointBinding {
  bodyId: number;
  mocapId: number;
  geomIds: number[];
}

interface HandBinding {
  /** One slot per `HAND_JOINTS`; null when the model lacks that body. */
  joints: (JointBinding | null)[];
  weldId: number;
  grabbed: number | null;
  collisions: boolean;
  /** Last pose written, per joint: where this control step's interpolation starts. */
  written: (HandJointPose | null)[];
  target: (HandJointPose | null)[];
  targetTracked: boolean;
}

function clonePose(pose: HandJointPose): HandJointPose {
  return { pos: [...pose.pos], quat: [...pose.quat] };
}

/**
 * Normalized lerp rather than a true slerp: consecutive XR frames are ~13 ms apart, so
 * the interpolated arc is small enough that the two agree to well under a pixel.
 */
function nlerpQuat(
  a: readonly number[],
  b: readonly number[],
  t: number
): [number, number, number, number] {
  const sign = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0 ? -1 : 1;
  const out = normalizeQuat([
    a[0] + (b[0] * sign - a[0]) * t,
    a[1] + (b[1] * sign - a[1]) * t,
    a[2] + (b[2] * sign - a[2]) * t,
    a[3] + (b[3] * sign - a[3]) * t,
  ]);
  return [out[0], out[1], out[2], out[3]];
}

export class HandMocapBinding {
  private readonly hands: Record<Handedness, HandBinding>;
  /** Every injected hand body, so a grab never targets the hand itself. */
  readonly bodyIds: Set<number>;

  private constructor(
    private readonly neqData: number,
    hands: Record<Handedness, HandBinding>,
    bodyIds: Set<number>
  ) {
    this.hands = hands;
    this.bodyIds = bodyIds;
  }

  /**
   * Resolve the injected rig in a compiled model, or return null when it is not there —
   * an un-injected scene, or one whose XML the splice could not reach.
   */
  static bind(mujoco: MainModule, mjModel: MjModel): HandMocapBinding | null {
    const bodyIdByName = new Map<string, number>();
    for (let b = 0; b < mjModel.nbody; b++) {
      bodyIdByName.set(mjModel.body(b).name, b);
    }
    const weldIdByName = new Map<string, number>();
    for (let e = 0; e < mjModel.neq; e++) {
      weldIdByName.set(mjModel.eq(e).name, e);
    }
    const geomsByBody = new Map<number, number[]>();
    for (let g = 0; g < mjModel.ngeom; g++) {
      const body = mjModel.geom_bodyid[g] as number;
      const list = geomsByBody.get(body);
      if (list) list.push(g);
      else geomsByBody.set(body, [g]);
    }

    const bodyIds = new Set<number>();
    const hands = {} as Record<Handedness, HandBinding>;
    for (const hand of HANDEDNESS) {
      const weldId = weldIdByName.get(handWeldName(hand));
      const joints = HAND_JOINTS.map(({ joint }) => {
        const bodyId = bodyIdByName.get(handBodyName(hand, joint));
        if (bodyId === undefined) return null;
        const mocapId = mjModel.body_mocapid[bodyId] as number;
        if (mocapId < 0) return null;
        bodyIds.add(bodyId);
        return { bodyId, mocapId, geomIds: geomsByBody.get(bodyId) ?? [] };
      });
      if (weldId === undefined || joints[PALM_INDEX] === null) {
        return null;
      }
      hands[hand] = {
        joints,
        weldId,
        grabbed: null,
        // Compiled collidable, so binding has to switch it off until a hand shows up.
        collisions: true,
        written: HAND_JOINTS.map(() => null),
        target: HAND_JOINTS.map(() => null),
        targetTracked: false,
      };
    }
    const binding = new HandMocapBinding(mujoco.mjNEQDATA, hands, bodyIds);
    for (const hand of HANDEDNESS) {
      binding.setCollisions(mjModel, hands[hand], false);
    }
    return binding;
  }

  /**
   * Latch the poses this control step interpolates towards. Called once per step, not per
   * substep: the snapshot the render loop publishes is the target, and what was written
   * last is the start, so MuJoCo never sees the hand jump a whole control step at once.
   */
  beginStep(frames: HandFrames): void {
    for (const hand of HANDEDNESS) {
      const binding = this.hands[hand];
      const frame = frames[hand];
      binding.targetTracked = frame.tracked;
      for (let i = 0; i < binding.target.length; i++) {
        const pose = frame.tracked ? frame.joints[i] : null;
        binding.target[i] = pose ? clonePose(pose) : null;
      }
    }
  }

  /** Write the latched poses at `alpha` of the way through this control step. */
  writeSubstep(mjModel: MjModel, mjData: MjData, alpha: number): void {
    for (const hand of HANDEDNESS) {
      const binding = this.hands[hand];
      this.setCollisions(mjModel, binding, binding.targetTracked);
      for (let i = 0; i < binding.joints.length; i++) {
        const joint = binding.joints[i];
        if (!joint) continue;
        const target = binding.target[i];
        if (!target) {
          this.writePose(mjData, joint, { pos: PARK_POS, quat: [1, 0, 0, 0] });
          binding.written[i] = null;
          continue;
        }
        const start = binding.written[i];
        const pose: HandJointPose = start
          ? {
              pos: [
                start.pos[0] + (target.pos[0] - start.pos[0]) * alpha,
                start.pos[1] + (target.pos[1] - start.pos[1]) * alpha,
                start.pos[2] + (target.pos[2] - start.pos[2]) * alpha,
              ],
              quat: nlerpQuat(start.quat, target.quat, alpha),
            }
          : target;
        this.writePose(mjData, joint, pose);
        if (alpha >= 1) binding.written[i] = clonePose(target);
      }
    }
  }

  /**
   * Start and stop pinch grabs. Runs from the step loop, not from the pinch event, so the
   * weld's relative pose is computed against the state MuJoCo is about to integrate.
   */
  applyGrabs(
    mjModel: MjModel,
    mjData: MjData,
    frames: HandFrames,
    graspable: Set<number> | null
  ): void {
    for (const hand of HANDEDNESS) {
      const binding = this.hands[hand];
      const frame = frames[hand];
      const holding = frame.tracked && frame.pinching;
      if (!holding) {
        if (binding.grabbed !== null) this.release(mjData, binding);
        continue;
      }
      if (binding.grabbed !== null) continue;
      const target = this.pickTarget(mjModel, mjData, frame, graspable);
      if (target !== null) this.grab(mjModel, mjData, binding, target);
    }
  }

  /** Drop every grab and forget the written poses; for a reset or a scene teardown. */
  reset(mjModel: MjModel, mjData: MjData): void {
    for (const hand of HANDEDNESS) {
      const binding = this.hands[hand];
      if (binding.grabbed !== null) this.release(mjData, binding);
      binding.written.fill(null);
      binding.target.fill(null);
      binding.targetTracked = false;
      // Written, not just forgotten: `mj_resetData` restores mjData, and contype lives
      // in mjModel, so a parked hand would otherwise keep the contacts it was enabled with.
      this.setCollisions(mjModel, binding, false);
    }
  }

  /** The body this hand is holding, or null. */
  grabbed(hand: Handedness): number | null {
    return this.hands[hand].grabbed;
  }

  private writePose(
    mjData: MjData,
    joint: JointBinding,
    pose: { pos: readonly number[]; quat: readonly number[] }
  ): void {
    const p = joint.mocapId * 3;
    mjData.mocap_pos[p + 0] = pose.pos[0];
    mjData.mocap_pos[p + 1] = pose.pos[1];
    mjData.mocap_pos[p + 2] = pose.pos[2];
    const q = joint.mocapId * 4;
    mjData.mocap_quat[q + 0] = pose.quat[0];
    mjData.mocap_quat[q + 1] = pose.quat[1];
    mjData.mocap_quat[q + 2] = pose.quat[2];
    mjData.mocap_quat[q + 3] = pose.quat[3];
  }

  /**
   * Contacts follow tracking: a hand the runtime is not reporting is parked, and a parked
   * hand must not obstruct the scene. Both levels are written because MuJoCo prunes
   * broadphase on the per-body aggregate the compiler derives from the geoms — writing
   * only `geom_contype` leaves the body pruned and the hand ghostly.
   */
  private setCollisions(mjModel: MjModel, binding: HandBinding, on: boolean): void {
    if (binding.collisions === on) return;
    binding.collisions = on;
    const value = on ? 1 : 0;
    for (const joint of binding.joints) {
      if (!joint) continue;
      mjModel.body_contype[joint.bodyId] = value;
      mjModel.body_conaffinity[joint.bodyId] = value;
      for (const g of joint.geomIds) {
        mjModel.geom_contype[g] = value;
        mjModel.geom_conaffinity[g] = value;
      }
    }
  }

  /** Nearest graspable body to the pinch point, by surface distance. */
  private pickTarget(
    mjModel: MjModel,
    mjData: MjData,
    frame: HandFrame,
    graspable: Set<number> | null
  ): number | null {
    const thumb = frame.joints[PINCH_INDICES[0]];
    const index = frame.joints[PINCH_INDICES[1]];
    if (!thumb || !index) return null;
    const point = [
      (thumb.pos[0] + index.pos[0]) / 2,
      (thumb.pos[1] + index.pos[1]) / 2,
      (thumb.pos[2] + index.pos[2]) / 2,
    ];

    let best: number | null = null;
    let bestDistance = GRAB_REACH;
    for (let g = 0; g < mjModel.ngeom; g++) {
      const body = mjModel.geom_bodyid[g] as number;
      if (body <= 0 || this.bodyIds.has(body)) continue;
      if (graspable && !graspable.has(body)) continue;
      const distance =
        Math.hypot(
          mjData.geom_xpos[g * 3 + 0] - point[0],
          mjData.geom_xpos[g * 3 + 1] - point[1],
          mjData.geom_xpos[g * 3 + 2] - point[2]
        ) - (mjModel.geom_rbound[g] as number);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = body;
      }
    }
    return best;
  }

  /**
   * Weld the grabbed body to the palm at the pose it is already in, so activating the
   * constraint holds the object instead of snapping it. `eq_data` for a weld is
   * `[anchor(3), relpose(3 pos + 4 quat), torquescale(1)]`, and its relpose is body2
   * expressed in body1's frame — body1 is the palm here.
   */
  private grab(
    mjModel: MjModel,
    mjData: MjData,
    binding: HandBinding,
    target: number
  ): void {
    const palm = binding.joints[PALM_INDEX];
    if (!palm) return;
    const palmPos = [
      mjData.xpos[palm.bodyId * 3 + 0],
      mjData.xpos[palm.bodyId * 3 + 1],
      mjData.xpos[palm.bodyId * 3 + 2],
    ];
    const palmQuat = [
      mjData.xquat[palm.bodyId * 4 + 0],
      mjData.xquat[palm.bodyId * 4 + 1],
      mjData.xquat[palm.bodyId * 4 + 2],
      mjData.xquat[palm.bodyId * 4 + 3],
    ];
    const targetQuat = [
      mjData.xquat[target * 4 + 0],
      mjData.xquat[target * 4 + 1],
      mjData.xquat[target * 4 + 2],
      mjData.xquat[target * 4 + 3],
    ];
    const relPos = quatApplyInv(palmQuat, [
      mjData.xpos[target * 3 + 0] - palmPos[0],
      mjData.xpos[target * 3 + 1] - palmPos[1],
      mjData.xpos[target * 3 + 2] - palmPos[2],
    ]);
    const relQuat = quatMultiply(quatInverse(palmQuat), targetQuat);

    const at = binding.weldId * this.neqData;
    mjModel.eq_data[at + 0] = 0;
    mjModel.eq_data[at + 1] = 0;
    mjModel.eq_data[at + 2] = 0;
    for (let i = 0; i < 3; i++) mjModel.eq_data[at + 3 + i] = relPos[i];
    for (let i = 0; i < 4; i++) mjModel.eq_data[at + 6 + i] = relQuat[i];
    mjModel.eq_data[at + 10] = 1;
    mjModel.eq_obj1id[binding.weldId] = palm.bodyId;
    mjModel.eq_obj2id[binding.weldId] = target;
    mjData.eq_active[binding.weldId] = 1;
    binding.grabbed = target;
  }

  private release(mjData: MjData, binding: HandBinding): void {
    mjData.eq_active[binding.weldId] = 0;
    binding.grabbed = null;
  }
}
