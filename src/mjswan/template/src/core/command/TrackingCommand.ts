import * as THREE from 'three';

import { quatApply, quatApplyInv, quatInverse, quatMultiply, yawQuat } from '../observation/math';
import { getPosition, getQuaternion } from '../scene/scene';
import { type NpzEntry, loadNpz } from '../scene/npz';
import { type Bytes, resolveBytes } from '../utils/bytes';
import { OnnxEvent, isOnnxEventConfig } from '../event/OnnxEvent';
import { LiveMotionSource, type LiveMotionStreamConfig, resolveStreamConfig } from './liveMotion';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type TrackingMotionConfig = {
  name: string;
  /** Raw `.npz` bytes (or a lazy loader) supplied by the app. */
  data: Bytes;
  fps: number;
  anchor_body_name: string;
  body_names: string[];
  dataset_joint_names?: string[];
  default?: boolean;
  loop?: boolean;
  clip_format?: 'body_world' | 'qpos';
  time_source?: 'wall' | 'sim';
  /** Free-form extras from the build; `stream` turns this motion into a live one. */
  metadata?: { stream?: LiveMotionStreamConfig } & Record<string, unknown>;
};

type LoadedTrackingMotion = TrackingMotionConfig & {
  jointPos: Float32Array[];
  jointVel: Float32Array[];
  bodyPosW: Float32Array[];
  bodyQuatW: Float32Array[];
  bodyLinVelW: Float32Array[];
  bodyAngVelW: Float32Array[];
  qposFrames?: Float32Array[];
  frameCount: number;
};

/**
 * Control steps between pose reports on a live clip.
 *
 * Every other step: the generator runs at about half the control rate, so this gives it a fresh
 * pose for each frame it produces without flooding the socket.
 */
const LIVE_CONTEXT_REPORT_EVERY = 2;

function normalizeQuat(quat: ArrayLike<number>): Float32Array {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return new Float32Array([
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ]);
}

function splitFrames(entry: NpzEntry): Float32Array[] {
  const totalFrames = entry.shape[0] ?? 0;
  const width = entry.shape.length <= 1 ? 1 : entry.shape.slice(1).reduce((acc, v) => acc * v, 1);
  const frames: Float32Array[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const out = new Float32Array(width);
    const start = i * width;
    for (let j = 0; j < width; j++) {
      out[j] = entry.data[start + j] ?? 0.0;
    }
    frames.push(out);
  }
  return frames;
}

function setGhostMaterial(material: THREE.Material): THREE.Material {
  const next = material.clone();
  if ('transparent' in next) {
    next.transparent = true;
  }
  if ('opacity' in next) {
    next.opacity = 0.5;
  }
  if ('depthWrite' in next) {
    next.depthWrite = false;
  }
  if ('color' in next && next.color instanceof THREE.Color) {
    next.color = new THREE.Color(0.5, 0.7, 0.5);
  }
  return next;
}

function hasRenderableMesh(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      found = true;
    }
  });
  return found;
}

/**
 * mjlab's `update_relative_body_poses`: place the reference bodies onto the robot by
 * keeping the robot's anchor x/y (but the *reference's* z) and rotating the
 * anchor-relative offsets by the yaw between the two anchors.
 */
export function reanchorBodyPositions(
  bodyPosW: Float32Array,
  anchorPos: ArrayLike<number>,
  anchorQuat: ArrayLike<number>,
  robotAnchorPos: ArrayLike<number>,
  robotAnchorQuat: ArrayLike<number>,
): Float32Array {
  const deltaPos = [robotAnchorPos[0] ?? 0, robotAnchorPos[1] ?? 0, anchorPos[2] ?? 0];
  const deltaOri = yawQuat(quatMultiply(robotAnchorQuat, quatInverse(anchorQuat)));
  const out = new Float32Array(bodyPosW.length);
  for (let i = 0; i + 2 < bodyPosW.length; i += 3) {
    const offset = quatApply(deltaOri, [
      bodyPosW[i] - (anchorPos[0] ?? 0),
      bodyPosW[i + 1] - (anchorPos[1] ?? 0),
      bodyPosW[i + 2] - (anchorPos[2] ?? 0),
    ]);
    for (let j = 0; j < 3; j++) {
      out[i + j] = deltaPos[j] + offset[j];
    }
  }
  return out;
}

export class TrackingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly motions: TrackingMotionConfig[];
  private readonly loadedMotions: Map<string, LoadedTrackingMotion>;
  private sampleHz: number;
  private readonly ghostRoot: THREE.Group | null;
  private readonly ghostBodies: Map<number, THREE.Group>;
  /** Model body id by name, filled on demand (see `resolveBodyId`). */
  private readonly bodyIds = new Map<string, number>();
  private readonly ghostData: import('mujoco').MjData | null;
  private refBodyPosW: Float32Array[];
  private refBodyQuatW: Float32Array[];
  private refBodyLinVelW: Float32Array[];
  private refBodyAngVelW: Float32Array[];
  private selectedMotionName: string | null;
  private selectedMotion: LoadedTrackingMotion | null;
  private selectedAnchorBodyIndex: number;
  private selectedRootBodyIndex: number;
  private datasetQposAdr: number[];
  private frameAccumulator: number;
  private justReset: boolean;
  private referenceVisible: boolean;
  private readonly samplingMode: string;
  /** Look-ahead/look-back offsets the `ref_*` window state fields are sampled at. */
  private readonly timeSteps: number[];
  /** Traced reference-state-initialization jitter, or null when the task jitters nothing. */
  private readonly resetJitter: OnnxEvent | null;
  refJointPos: Float32Array[];
  refJointVel: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  nJoints: number;
  /** Set when the selected motion streams its frames instead of shipping them. */
  private liveSource: LiveMotionSource | null;
  /** Whether the streamed frames have replaced the bundled ones. */
  private liveAdopted: boolean;
  /** Streamed frames already mirrored into the derived root arrays. */
  private liveMirrored: number;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    context: CommandTermContext,
  ) {
    this.context = context;
    this.motions = Array.isArray(config.motions) ? config.motions as TrackingMotionConfig[] : [];
    this.loadedMotions = new Map();
    this.sampleHz = 50.0;
    this.selectedMotionName =
      this.motions.find((motion) => motion.default)?.name ??
      this.motions[0]?.name ??
      null;
    this.selectedMotion = null;
    this.selectedAnchorBodyIndex = 0;
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = [];
    this.frameAccumulator = 0.0;
    this.justReset = true;
    this.referenceVisible = true;
    this.samplingMode = typeof config.sampling_mode === 'string' ? config.sampling_mode : 'start';
    this.timeSteps = Array.isArray(config.time_steps)
      ? (config.time_steps as unknown[]).map((step) => Math.trunc(Number(step) || 0))
      : [0];
    this.resetJitter = this.buildResetJitter(config.reset_graph);
    this.refJointPos = [];
    this.refJointVel = [];
    this.refRootPos = [];
    this.refRootQuat = [];
    this.refIdx = 0;
    this.refLen = 0;
    this.nJoints = this.motions.find((motion) => motion.name === this.selectedMotionName)?.dataset_joint_names?.length ?? 0;

    this.ghostBodies = new Map();
    this.ghostData = context.mjModel ? new context.mujoco.MjData(context.mjModel) : null;
    this.ghostRoot = this.createGhostRoot();
    this.refBodyPosW = [];
    this.refBodyQuatW = [];
    this.refBodyLinVelW = [];
    this.refBodyAngVelW = [];
    this.liveSource = null;
    this.liveAdopted = false;
    this.liveMirrored = 0;
  }

  getCommand(): Float32Array {
    if (!this.selectedMotion || this.refLen === 0) {
      return new Float32Array(this.nJoints * 2);
    }
    if (this.selectedMotion.clip_format === 'qpos') {
      return new Float32Array(0);
    }
    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(this.nJoints);
    const jointVel = this.selectedMotion.jointVel[this.refIdx] ?? new Float32Array(this.nJoints);
    const out = new Float32Array(jointPos.length + jointVel.length);
    out.set(jointPos, 0);
    out.set(jointVel, jointPos.length);
    return out;
  }

  getUiConfig(): CommandUiConfig | null {
    return null;
  }

  async setSelectedMotion(name: string | null): Promise<boolean> {
    if (name === null) {
      this.selectedMotionName = null;
      this.selectedMotion = null;
      this.refJointPos = [];
      this.refJointVel = [];
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      this.refLen = 0;
      this.nJoints = 0;
      this.updateGhostPose();
      return false;
    }

    const config = this.motions.find((motion) => motion.name === name);
    if (!config) {
      return false;
    }

    const loaded = this.loadedMotions.get(name) ?? await this.loadMotion(config);
    this.loadedMotions.set(name, loaded);
    this.selectedMotionName = name;
    this.selectedMotion = loaded;
    this.selectedAnchorBodyIndex = Math.max(
      0,
      loaded.body_names.indexOf(loaded.anchor_body_name),
    );
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = this.resolveQposAdr(loaded.dataset_joint_names ?? []);
    this.refLen = loaded.frameCount;
    this.refJointPos = loaded.jointPos;
    this.refJointVel = loaded.jointVel;
    this.refIdx = this.sampleInitialFrame(this.refLen);
    this.nJoints = loaded.jointPos[0]?.length ?? 0;
    this.frameAccumulator = 0.0;
    this.justReset = true;
    this.updateReferenceState();
    this.applyReferenceStateToSim();
    this.updateGhostPose();
    return true;
  }

  setReferenceVisible(visible: boolean): void {
    this.referenceVisible = visible;
    if (this.ghostRoot) {
      this.ghostRoot.visible = visible && this.selectedMotion !== null;
    }
  }

  reset(): void {
    // A live clip has no beginning to go back to: frame 0 is wherever the session started, minutes
    // and metres ago. Resuming there would teleport the robot to the origin and hand it a reference
    // it has already walked through, so an episode reset lands on the newest frames instead.
    this.refIdx = this.liveAdopted ? this.liveFrontier() : this.sampleInitialFrame(this.refLen);
    this.frameAccumulator = 0.0;
    this.justReset = true;
    this.updateReferenceState();
    this.applyReferenceStateToSim();
    this.updateGhostPose();
  }

  update(dt: number): void {
    if (this.liveSource) {
      if (!this.liveAdopted) {
        if (this.liveSource.length > 0) {
          this.adoptLiveFrames();
        }
      } else {
        this.syncLiveFrames();
        // Ask for frames past the furthest look-ahead the policy reads, not just past the cursor.
        this.liveSource.ensure(this.refIdx + Math.max(0, ...this.timeSteps));
        this.reportLiveContext();
      }
    }
    if (!this.selectedMotion || this.refLen <= 1) {
      return;
    }
    if (this.justReset) {
      this.justReset = false;
      this.updateGhostPose();
      return;
    }
    if (this.selectedMotion.time_source === 'sim') {
      const simTime = this.context.mjData?.time ?? 0;
      this.refIdx = this.sampleHz > 0 ? Math.floor(simTime * this.sampleHz) % this.refLen : 0;
      this.updateGhostPose();
      return;
    }
    // A live clip's end is the generator running behind, not the motion being over: holding the
    // last frame lets it catch up, whereas looping would restart the episode every time it does.
    const shouldLoop = this.liveAdopted ? false : this.selectedMotion?.loop !== false;
    this.frameAccumulator += dt * this.sampleHz;
    let motionLooped = false;
    while (this.frameAccumulator >= 1.0) {
      this.refIdx += 1;
      if (this.refIdx >= this.refLen) {
        if (!shouldLoop) {
          this.refIdx = this.refLen - 1;
          this.frameAccumulator = 0.0;
          break;
        }
        this.refIdx = 0;
        motionLooped = true;
      }
      this.frameAccumulator -= 1.0;
    }
    if (motionLooped) {
      this.context.requestReset?.();
    }
    this.updateGhostPose();
  }

  updateDebugVisuals(): void {
    if (this.ghostRoot) {
      this.ghostRoot.visible = this.referenceVisible && this.selectedMotion !== null;
    }
  }

  dispose(): void {
    if (this.ghostRoot) {
      this.ghostRoot.parent?.remove(this.ghostRoot);
      this.ghostRoot.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            for (const material of obj.material) {
              material.dispose?.();
            }
          } else {
            obj.material?.dispose?.();
          }
        }
      });
    }
    this.ghostData?.delete?.();
  }

  isReady(): boolean {
    return this.selectedMotion !== null && this.refLen > 0;
  }

  getSelectedMotion(): LoadedTrackingMotion | null {
    return this.selectedMotion;
  }

  getSelectedMotionName(): string | null {
    return this.selectedMotionName;
  }

  getAnchorBodyName(): string | null {
    return this.selectedMotion?.anchor_body_name
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.anchor_body_name
      ?? null;
  }

  getBodyNames(): string[] {
    return this.selectedMotion?.body_names
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.body_names
      ?? [];
  }

  getAnchorBodyIndex(): number {
    return this.selectedAnchorBodyIndex;
  }

  getAnchorPos(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 3;
    return frame.slice(offset, offset + 3);
  }

  getAnchorQuat(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyQuatW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 4;
    return normalizeQuat(frame.slice(offset, offset + 4));
  }

  /** The anchor body's slice of a per-body world vector field, at the current frame. */
  private anchorVector(frames: Float32Array[]): Float32Array | null {
    const frame = frames[this.refIdx];
    if (!frame) return null;
    const offset = this.selectedAnchorBodyIndex * 3;
    return frame.slice(offset, offset + 3);
  }

  /** {@link anchorVector}, rotated into the anchor's own frame. */
  private anchorFrameVector(frames: Float32Array[]): Float32Array | null {
    const vector = this.anchorVector(frames);
    const anchorQuat = this.getAnchorQuat();
    if (!vector || !anchorQuat) return null;
    return Float32Array.from(quatApplyInv(anchorQuat, vector));
  }

  getBodyPosW(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    return frame ? frame.slice() : null;
  }

  /**
   * mjlab `MotionCommand` state, for traced graphs declaring a `{command: "motion",
   * field}` slot — in mjlab's frame, element order and units (`env_origins` omitted,
   * since the browser runs one env at the origin). An unlisted field returns null and
   * its caller holds the previous value.
   *
   * The `ref_*` fields and `is_ready` are the look-ahead window, which mjlab has no
   * equivalent of: each is the `time_steps` offsets' frames concatenated, and the
   * traced term slices out the ones it wants.
   */
  getStateField(field: string): Float32Array | null {
    switch (field) {
      case 'is_ready':
        return new Float32Array([this.isReady() ? 1.0 : 0.0]);
      case 'ref_root_pos_w':
        return this.refWindow(this.refRootPos, 3);
      case 'ref_root_quat_w':
        return this.refWindow(this.refRootQuat, 4, true);
      case 'ref_joint_pos':
        return this.refWindow(this.refJointPos, this.nJoints);
      case 'ref_joint_vel':
        return this.refWindow(this.refJointVel, this.nJoints);
      // Whole-body reference over the window, for tasks whose goal is Cartesian keypoints
      // rather than the root alone. Anchor- and root-only channels are a slice of these.
      case 'ref_body_pos_w':
        return this.refWindow(this.refBodyPosW, this.getBodyNames().length * 3);
      case 'ref_body_quat_w':
        return this.refBodyQuatWindow();
      case 'anchor_pos_w':
        return this.getAnchorPos();
      case 'anchor_quat_w':
        return this.getAnchorQuat();
      case 'anchor_lin_vel_w':
        return this.anchorVector(this.refBodyLinVelW);
      case 'anchor_ang_vel_w':
        return this.anchorVector(this.refBodyAngVelW);
      // Anchor-frame reference features: the world quantities above rotated into the
      // anchor's frame, as mjlab's `quat_apply_inverse(anchor_quat_w, …)` properties do.
      case 'ref_base_height': {
        // `env_origins` is omitted throughout: the browser runs one env at the origin.
        const anchorPos = this.getAnchorPos();
        return anchorPos ? new Float32Array([anchorPos[2]]) : null;
      }
      case 'ref_base_lin_vel_b':
        return this.anchorFrameVector(this.refBodyLinVelW);
      case 'ref_base_ang_vel_b':
        return this.anchorFrameVector(this.refBodyAngVelW);
      case 'ref_gravity_b': {
        const anchorQuat = this.getAnchorQuat();
        return anchorQuat
          ? Float32Array.from(quatApplyInv(anchorQuat, [0, 0, -1]))
          : null;
      }
      case 'joint_pos':
      case 'tracked_joint_pos':
        // The current reference frame alone, where `ref_joint_pos` is the whole
        // look-ahead window.
        return this.refJointPos[this.refIdx]?.slice() ?? null;
      case 'body_pos_w':
        return this.getBodyPosW();
      case 'robot_anchor_pos_w':
        return this.robotBodyField('xpos', 3, [this.getAnchorBodyName() ?? '']);
      case 'robot_anchor_quat_w':
        return this.robotBodyField('xquat', 4, [this.getAnchorBodyName() ?? '']);
      case 'robot_body_pos_w':
        return this.robotBodyField('xpos', 3, this.getBodyNames());
      case 'body_pos_relative_w':
        return this.bodyPosRelativeW();
      default:
        return null;
    }
  }

  /**
   * One `ref_*` field at every `time_steps` offset, concatenated. Offsets clamp rather
   * than wrap, as in training; not ready gives zeros the `is_ready` gate multiplies away.
   */
  private refWindow(frames: Float32Array[], stride: number, quat = false): Float32Array {
    const out = new Float32Array(this.timeSteps.length * stride);
    if (quat) {
      for (let i = 0; i < this.timeSteps.length; i++) out[i * stride] = 1.0;
    }
    if (!this.isReady()) {
      return out;
    }
    for (let i = 0; i < this.timeSteps.length; i++) {
      const index = Math.min(this.refLen - 1, Math.max(0, this.refIdx + this.timeSteps[i]));
      const frame = frames[index];
      if (!frame) continue;
      const values = quat ? normalizeQuat(frame) : frame;
      for (let j = 0; j < stride && j < values.length; j++) {
        out[i * stride + j] = values[j];
      }
    }
    return out;
  }

  /**
   * {@link refWindow} for a per-body quaternion field: every body normalized, and the
   * not-ready fill an identity per body rather than one identity and zeros.
   */
  private refBodyQuatWindow(): Float32Array {
    const nBodies = this.getBodyNames().length;
    const stride = nBodies * 4;
    const out = new Float32Array(this.timeSteps.length * stride);
    for (let i = 0; i < this.timeSteps.length; i++) {
      for (let b = 0; b < nBodies; b++) out[i * stride + b * 4] = 1.0;
    }
    if (!this.isReady()) {
      return out;
    }
    for (let i = 0; i < this.timeSteps.length; i++) {
      const index = Math.min(this.refLen - 1, Math.max(0, this.refIdx + this.timeSteps[i]));
      const frame = this.refBodyQuatW[index];
      if (!frame) continue;
      for (let b = 0; b < nBodies && (b + 1) * 4 <= frame.length; b++) {
        out.set(normalizeQuat(frame.subarray(b * 4, b * 4 + 4)), i * stride + b * 4);
      }
    }
    return out;
  }

  /** `mjData.<source>` rows for the named bodies, flattened — mjlab's `body_link_*_w`. */
  private robotBodyField(
    source: 'xpos' | 'xquat',
    stride: number,
    bodyNames: string[],
  ): Float32Array | null {
    const mjData = this.context.mjData;
    if (!mjData || bodyNames.length === 0) {
      return null;
    }
    const out = new Float32Array(bodyNames.length * stride);
    for (let i = 0; i < bodyNames.length; i++) {
      const bodyId = this.resolveBodyId(bodyNames[i]);
      if (bodyId < 0) {
        return null;
      }
      for (let j = 0; j < stride; j++) {
        out[i * stride + j] = mjData[source][bodyId * stride + j] ?? 0.0;
      }
    }
    return out;
  }

  /** The reference bodies re-anchored onto the robot (`reanchorBodyPositions`). */
  private bodyPosRelativeW(): Float32Array | null {
    const anchorPos = this.getAnchorPos();
    const anchorQuat = this.getAnchorQuat();
    const bodyPos = this.getBodyPosW();
    const anchorBody = [this.getAnchorBodyName() ?? ''];
    const robotAnchorPos = this.robotBodyField('xpos', 3, anchorBody);
    const robotAnchorQuat = this.robotBodyField('xquat', 4, anchorBody);
    if (!anchorPos || !anchorQuat || !bodyPos || !robotAnchorPos || !robotAnchorQuat) {
      return null;
    }
    return reanchorBodyPositions(bodyPos, anchorPos, anchorQuat, robotAnchorPos, robotAnchorQuat);
  }

  /** `findBodyIdByName`, memoized: the slots are read every control step. */
  private resolveBodyId(bodyName: string): number {
    let bodyId = this.bodyIds.get(bodyName);
    if (bodyId === undefined) {
      bodyId = this.findBodyIdByName(bodyName);
      this.bodyIds.set(bodyName, bodyId);
    }
    return bodyId;
  }

  private createGhostRoot(): THREE.Group | null {
    const bodies = this.context.bodies ?? null;
    const mjModel = this.context.mjModel;
    if (!bodies || !mjModel) {
      return null;
    }
    const root = new THREE.Group();
    root.name = 'Tracking Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
      const numericBodyId = Number(bodyId);
      if (!this.isDynamicBody(numericBodyId)) {
        continue;
      }
      const clone = body.clone(true);
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            obj.material = obj.material.map(setGhostMaterial);
          } else {
            obj.material = setGhostMaterial(obj.material);
          }
          obj.renderOrder = 2;
        }
      });
      if (!hasRenderableMesh(clone)) {
        continue;
      }
      this.ghostBodies.set(numericBodyId, clone);
      root.add(clone);
    }
    (this.context.mujocoRoot ?? this.context.scene).add(root);
    return root;
  }

  private isDynamicBody(bodyId: number): boolean {
    const mjModel = this.context.mjModel;
    if (!mjModel || bodyId <= 0 || bodyId >= mjModel.nbody) {
      return false;
    }

    let current = bodyId;
    while (current > 0) {
      if (mjModel.body_jntnum[current] > 0) {
        return true;
      }
      current = mjModel.body_parentid[current];
    }
    return false;
  }

  private async loadMotion(config: TrackingMotionConfig): Promise<LoadedTrackingMotion> {
    this.sampleHz = config.fps;
    const stream = await resolveStreamConfig(config.metadata?.stream);
    if (stream) {
      // Connect now and keep playing the bundled clip meanwhile: the generator needs a moment to
      // produce its opening frames, and a clip that already tracks is a better thing to show than
      // a robot with no reference at all. `adoptLiveFrames` switches over once frames exist.
      this.liveSource = new LiveMotionSource(stream);
      this.liveSource.connect();
    }
    const npz = await loadNpz(await resolveBytes(config.data));
    const empty: Float32Array[] = [];

    if (config.clip_format === 'qpos') {
      if (!npz['qpos']) {
        throw new Error("Motion asset with clip_format='qpos' is missing 'qpos'");
      }
      const qposFrames = splitFrames(npz['qpos']!);
      return {
        ...config,
        jointPos: empty,
        jointVel: empty,
        bodyPosW: empty,
        bodyQuatW: empty,
        bodyLinVelW: empty,
        bodyAngVelW: empty,
        qposFrames,
        frameCount: qposFrames.length,
      };
    }

    const required = ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'] as const;
    for (const key of required) {
      if (!npz[key]) {
        throw new Error(`Motion asset is missing '${key}'`);
      }
    }
    const jointPos = splitFrames(npz['joint_pos']!);
    const jointVel = splitFrames(npz['joint_vel']!);
    const sourceBodyNames = npz['body_names']?.strings ?? null;
    const bodyPosW = this.selectMotionBodyFrames(splitFrames(npz['body_pos_w']!), config.body_names, 3, sourceBodyNames);
    const bodyQuatW = this.selectMotionBodyFrames(splitFrames(npz['body_quat_w']!), config.body_names, 4, sourceBodyNames);
    const bodyLinVelW = this.selectMotionBodyFrames(splitFrames(npz['body_lin_vel_w']!), config.body_names, 3, sourceBodyNames);
    const bodyAngVelW = this.selectMotionBodyFrames(splitFrames(npz['body_ang_vel_w']!), config.body_names, 3, sourceBodyNames);
    return { ...config, jointPos, jointVel, bodyPosW, bodyQuatW, bodyLinVelW, bodyAngVelW, frameCount: jointPos.length };
  }

  private selectMotionBodyFrames(
    frames: Float32Array[],
    bodyNames: string[],
    stride: number,
    sourceBodyNames: string[] | null = null,
  ): Float32Array[] {
    const mjModel = this.context.mjModel;
    const first = frames[0];
    if (!mjModel || !first || bodyNames.length === 0) {
      return frames;
    }

    const sourceBodyCount = Math.floor(first.length / stride);
    if (sourceBodyCount === bodyNames.length) {
      return frames;
    }

    let bodySourceIndices: number[];
    if (sourceBodyNames !== null) {
      // Use the npz's own body-name manifest for unambiguous index lookup.
      bodySourceIndices = bodyNames.map((name) => sourceBodyNames.indexOf(name));
    } else {
      // Fall back to mjModel body-ID order from the first body in body_names.
      const rootBodyId = this.findBodyIdByName(bodyNames[0]);
      const bodyIds = bodyNames.map((name) => this.findBodyIdByName(name));
      bodySourceIndices = bodyIds.map((id) => id - rootBodyId);
    }

    if (bodySourceIndices.some((idx) => idx < 0 || idx >= sourceBodyCount)) {
      console.warn('[TrackingCommand] Could not map all motion body names to source body indices; using raw body frames.');
      return frames;
    }

    return frames.map((frame) => {
      const selected = new Float32Array(bodyNames.length * stride);
      for (let i = 0; i < bodySourceIndices.length; i++) {
        const sourceOffset = bodySourceIndices[i] * stride;
        const targetOffset = i * stride;
        for (let j = 0; j < stride; j++) {
          selected[targetOffset + j] = frame[sourceOffset + j] ?? 0.0;
        }
      }
      return selected;
    });
  }

  /**
   * Switch from the bundled clip to the streamed one.
   *
   * The two are unrelated trajectories in the same world, so this cannot be a seam: the frames are
   * replaced wholesale and an episode reset is requested, which re-places the robot on the new
   * frame 0 and re-primes the policy's history from that pose. Appending instead would teleport
   * the reference mid-stride.
   */
  private adoptLiveFrames(): void {
    const source = this.liveSource;
    const motion = this.selectedMotion;
    if (!source || !motion) {
      return;
    }
    // Shared array objects, not copies: the source appends to exactly these as blocks arrive.
    motion.jointPos = source.frames.jointPos;
    motion.jointVel = source.frames.jointVel;
    motion.bodyPosW = source.frames.bodyPosW;
    motion.bodyQuatW = source.frames.bodyQuatW;
    motion.bodyLinVelW = source.frames.bodyLinVelW;
    motion.bodyAngVelW = source.frames.bodyAngVelW;
    this.refJointPos = source.frames.jointPos;
    this.refJointVel = source.frames.jointVel;
    this.refBodyPosW = source.frames.bodyPosW;
    this.refBodyQuatW = source.frames.bodyQuatW;
    this.refBodyLinVelW = source.frames.bodyLinVelW;
    this.refBodyAngVelW = source.frames.bodyAngVelW;
    this.refRootPos = [];
    this.refRootQuat = [];
    this.liveMirrored = 0;
    this.liveAdopted = true;
    this.nJoints = source.frames.jointPos[0]?.length ?? this.nJoints;
    this.syncLiveFrames();
    // Start at the newest frames, not at the opening ones: the generator has been running since the
    // socket opened, and the frames it produced while the bundled clip played are already history.
    this.refIdx = this.liveFrontier();
    // The hand-over is the one moment the robot is teleported: the reset re-places it on the live
    // reference and re-primes the policy's history there. (Logging this is no help -- the
    // production build strips `console.*`.)
    this.context.requestReset?.();
  }

  /**
   * The newest frame a live clip can be read from, leaving the policy's look-ahead in hand.
   *
   * Reading here rather than from the back of the buffer keeps the correction loop short: the
   * generator's feedback is measured at the cursor, so a cursor far behind the frontier means it is
   * correcting against stale information.
   */
  private liveFrontier(): number {
    return Math.max(0, this.refLen - 1 - Math.max(0, ...this.timeSteps));
  }

  /**
   * Send the robot's pose upstream, so the generator can continue from where the robot actually is.
   *
   * Assembled in the contract's order -- the free joint's seven, then the dataset joints -- because
   * that is the layout the generator's own model uses, and the two are index-compatible.
   */
  private reportLiveContext(): void {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    if (!mjModel || !mjData || this.refIdx % LIVE_CONTEXT_REPORT_EVERY !== 0) {
      return;
    }
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex < 0) {
      return;
    }
    const qposAdr = mjModel.jnt_qposadr[freeJointIndex];
    const qpos = new Float64Array(7 + this.datasetQposAdr.length);
    for (let i = 0; i < 7; i++) {
      qpos[i] = mjData.qpos[qposAdr + i];
    }
    for (let i = 0; i < this.datasetQposAdr.length; i++) {
      qpos[7 + i] = mjData.qpos[this.datasetQposAdr[i]];
    }
    this.liveSource?.reportContext(qpos, this.refIdx, this.refRootQuat[this.refIdx] ?? undefined);
  }

  /**
   * Derive the root arrays for frames that arrived since the last call.
   *
   * The body arrays are shared with the source and need nothing done to them; root position and
   * orientation are slices out of each body frame, so they are built here -- incrementally, since
   * rebuilding the whole clip every time a block lands is quadratic in a session's length.
   */
  private syncLiveFrames(): void {
    const source = this.liveSource;
    if (!source) {
      return;
    }
    // Re-point at the stream's arrays every time, so nothing that rebuilt them can leave the
    // tracker reading a stale copy.
    this.refJointPos = source.frames.jointPos;
    this.refJointVel = source.frames.jointVel;
    this.refBodyPosW = source.frames.bodyPosW;
    this.refBodyQuatW = source.frames.bodyQuatW;
    this.refBodyLinVelW = source.frames.bodyLinVelW;
    this.refBodyAngVelW = source.frames.bodyAngVelW;
    for (let i = this.liveMirrored; i < source.length; i++) {
      const bodyPos = source.frames.bodyPosW[i];
      const bodyQuat = source.frames.bodyQuatW[i];
      if (!bodyPos || !bodyQuat) {
        break;
      }
      const root = this.selectedRootBodyIndex;
      this.refRootPos.push(bodyPos.slice(root * 3, root * 3 + 3));
      this.refRootQuat.push(normalizeQuat(bodyQuat.slice(root * 4, root * 4 + 4)));
      this.liveMirrored = i + 1;
    }
    this.refLen = this.liveMirrored;
  }

  private updateReferenceState(): void {
    // A live clip owns its arrays: they are the stream's own, and they grow as frames arrive.
    // Rebuilding them here as copies -- which an episode reset would otherwise do, including the
    // reset that hands over to the stream -- freezes them at whatever had arrived by then, while
    // `refLen` keeps climbing. Every windowed read then indexes past the end and the policy is fed
    // an all-zero reference: it walks on proprioception alone, never turning or backing up, while
    // the root arrays (pushed to directly) still look perfectly correct from outside.
    if (this.liveAdopted) {
      this.syncLiveFrames();
      return;
    }
    const motion = this.selectedMotion;
    if (!motion || motion.frameCount === 0 || motion.clip_format === 'qpos') {
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      return;
    }

    this.refBodyPosW = motion.bodyPosW.map((frame) => frame.slice());
    this.refBodyQuatW = motion.bodyQuatW.map((frame) => frame.slice());
    this.refBodyLinVelW = motion.bodyLinVelW.map((frame) => frame.slice());
    this.refBodyAngVelW = motion.bodyAngVelW.map((frame) => frame.slice());
    this.refRootPos = this.refBodyPosW.map((frame) =>
      frame.slice(this.selectedRootBodyIndex * 3, this.selectedRootBodyIndex * 3 + 3),
    );
    this.refRootQuat = this.refBodyQuatW.map((frame) =>
      normalizeQuat(frame.slice(this.selectedRootBodyIndex * 4, this.selectedRootBodyIndex * 4 + 4)),
    );
  }

  private applyReferenceStateToSim(): void {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    const motion = this.selectedMotion;
    if (!mjModel || !mjData || !motion || this.refLen === 0 || motion.clip_format === 'qpos') {
      return;
    }

    const rootPos = this.sampleRootPos(this.refIdx);
    const rootQuat = this.sampleRootQuat(this.refIdx);
    const freeJointIndex = this.findFreeJointIndex();
    if (rootPos && rootQuat && freeJointIndex >= 0) {
      const qposAdr = mjModel.jnt_qposadr[freeJointIndex];
      const qvelAdr = mjModel.jnt_dofadr[freeJointIndex];
      mjData.qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      mjData.qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      mjData.qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      mjData.qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      mjData.qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      mjData.qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      mjData.qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;

      const linVel = this.sampleRootVelocity(this.refIdx, this.refBodyLinVelW);
      const angVel = this.sampleRootAngularVelocity(this.refIdx);
      if (linVel && angVel) {
        mjData.qvel[qvelAdr + 0] = linVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 1] = linVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 2] = linVel[2] ?? 0.0;
        mjData.qvel[qvelAdr + 3] = angVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 4] = angVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 5] = angVel[2] ?? 0.0;
      }
    }

    const jointPos = this.sampleJointPos(this.refIdx);
    const jointVel = motion.jointVel[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      mjData.qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }
    for (let i = 0; i < this.datasetQposAdr.length && i < jointVel.length; i++) {
      const dofAdr = this.resolveQvelAdrForQposAdr(this.datasetQposAdr[i]);
      if (dofAdr >= 0) {
        mjData.qvel[dofAdr] = jointVel[i] ?? 0.0;
      }
    }

    this.context.mujoco.mj_forward(mjModel, mjData);
    this.applyResetJitter();
  }

  /**
   * Run the traced reference-state-initialization graph, if the build shipped one.
   *
   * mjlab perturbs the reference frame before writing it; this perturbs it after,
   * reading it back off `asset.data` — same numbers, and the clip stays out of the
   * graph. The `mj_forward` above is what makes that read valid.
   *
   * Fire-and-forget, since `reset()` is sync and ORT is not. A frame of un-jittered
   * reference pose is harmless.
   */
  private applyResetJitter(): void {
    const graph = this.resetJitter;
    if (!graph) return;
    void graph
      .fire({
        mjModel: this.context.mjModel,
        mjData: this.context.mjData,
        terrainData: null,
      })
      .then(() => {
        const { mjModel, mjData } = this.context;
        if (mjModel && mjData) this.context.mujoco.mj_forward(mjModel, mjData);
      });
  }

  /**
   * The RSI graph, run through `OnnxEvent` rather than a second `rand`+`entity_write`
   * evaluator. Skips if it or the PRNG is absent: a less varied start, not a broken one.
   */
  private buildResetJitter(config: unknown): OnnxEvent | null {
    if (!isOnnxEventConfig(config)) return null;
    const session = this.context.onnxSessions?.get(config.onnx);
    const rng = this.context.rng;
    if (!session || !rng) {
      console.warn(
        `[TrackingCommand] reset jitter needs the ONNX session "${config.onnx}" and a ` +
          'seeded rng; starting from the unjittered reference frame.',
      );
      return null;
    }
    return new OnnxEvent(config, { session, rng, readSlot: this.context.readOnnxSlot });
  }

  private sampleRootPos(frameIndex: number): Float32Array | null {
    return this.refRootPos[frameIndex] ?? null;
  }

  private sampleRootQuat(frameIndex: number): Float32Array | null {
    return this.refRootQuat[frameIndex] ?? null;
  }

  private sampleRootVelocity(frameIndex: number, source: Float32Array[]): Float32Array | null {
    return (
      source[frameIndex]?.slice(
        this.selectedRootBodyIndex * 3,
        this.selectedRootBodyIndex * 3 + 3,
      ) ?? null
    );
  }

  private sampleRootAngularVelocity(frameIndex: number): Float32Array | null {
    return this.sampleRootVelocity(frameIndex, this.refBodyAngVelW);
  }

  private sampleInitialFrame(frameCount: number): number {
    if (frameCount <= 1 || this.samplingMode === 'start') {
      return 0;
    }
    if (this.samplingMode === 'uniform') {
      // The seeded PRNG, not `Math.random()`, so a session replays; mjlab uses randint.
      const rng = this.context.rng;
      if (!rng) {
        console.warn('[TrackingCommand] no seeded rng in context; starting at frame 0.');
        return 0;
      }
      return Math.min(frameCount - 1, Math.floor(rng.next() * frameCount));
    }
    return 0;
  }

  private sampleJointPos(frameIndex: number): Float32Array {
    return this.refJointPos[frameIndex] ?? new Float32Array(0);
  }

  private resolveQposAdr(jointNames: string[]): number[] {
    const mjModel = this.context.mjModel;
    if (!mjModel || jointNames.length === 0) {
      return [];
    }
    const resolved: number[] = [];
    for (const jointName of jointNames) {
      let adr = -1;
      for (let j = 0; j < mjModel.njnt; j++) {
        const modelJointName = mjModel.jnt(j).name;
        if (modelJointName === jointName || modelJointName.endsWith(`/${jointName}`)) {
          adr = mjModel.jnt_qposadr[j];
          break;
        }
      }
      if (adr >= 0) {
        resolved.push(adr);
      }
    }
    return resolved;
  }

  private resolveQvelAdrForQposAdr(qposAdr: number): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let j = 0; j < mjModel.njnt; j++) {
      if (mjModel.jnt_qposadr[j] === qposAdr) {
        return mjModel.jnt_dofadr[j];
      }
    }
    return -1;
  }

  private findBodyIdByName(bodyName: string): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let b = 0; b < mjModel.nbody; b++) {
      const name = mjModel.body(b).name;
      if (name === bodyName || name.endsWith(`/${bodyName}`)) {
        return b;
      }
    }
    return -1;
  }

  private updateGhostPose(): void {
    if (!this.ghostRoot || !this.ghostData || !this.context.mjModel || !this.selectedMotion || !this.refLen) {
      if (this.ghostRoot) {
        this.ghostRoot.visible = false;
      }
      return;
    }

    if (this.selectedMotion.clip_format === 'qpos' && this.selectedMotion.qposFrames) {
      const frame = this.selectedMotion.qposFrames[this.refIdx];
      if (frame) {
        this.ghostData.qpos.set(frame);
      }
      this.context.mujoco.mj_forward(this.context.mjModel, this.ghostData);
      for (const [bodyId, body] of this.ghostBodies) {
        getPosition(this.ghostData.xpos, bodyId, body.position);
        getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
      }
      this.ghostRoot.visible = this.referenceVisible;
      return;
    }

    const qpos = this.ghostData.qpos;
    qpos.set(this.context.mjModel.qpos0);

    const rootPos = this.refRootPos[this.refIdx];
    const rootQuat = this.refRootQuat[this.refIdx];
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex >= 0 && rootPos && rootQuat) {
      const qposAdr = this.context.mjModel.jnt_qposadr[freeJointIndex];
      qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;
    }

    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }

    this.context.mujoco.mj_forward(this.context.mjModel, this.ghostData);

    for (const [bodyId, body] of this.ghostBodies) {
      getPosition(this.ghostData.xpos, bodyId, body.position);
      getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
    }
    this.ghostRoot.visible = this.referenceVisible;
  }

  private findFreeJointIndex(): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) {
        return i;
      }
    }
    return -1;
  }
}
