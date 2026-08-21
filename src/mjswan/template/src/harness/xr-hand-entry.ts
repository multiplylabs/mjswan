/**
 * Physics-tier E2E for VR hand tracking: drives the real hand rig against the real
 * MuJoCo wasm with a scripted hand, so the parts no unit test can reach — the injected
 * MJCF compiling, a mocap hand generating contacts, a pinch weld actually holding an
 * object, contacts staying off while a hand is untracked — are measured in a browser
 * instead of asserted in prose.
 *
 * No headset involved: `XrHandTracker` is the only piece that needs a live session, and
 * all it produces is a `HandFrames` snapshot, which this harness writes directly.
 */
import { stepPhysics } from '../core/action/applyAction';
import {
  createHandFrames,
  HandMocapBinding,
  type HandFrames,
  type HandJointPose,
} from '../core/xr/HandMocap';
import { HAND_JOINTS, injectHandRig, PALM_JOINT } from '../core/xr/handRig';

export interface XrHandOutcome {
  ok: boolean;
  error?: string;
  /** Injection compiled and every rig body/weld resolved. */
  bound?: boolean;
  bodies?: { total: number; mocap: number; equalities: number };
  /** Cube displacement (m) from a tracked hand sweeping into it. */
  pushed?: number;
  /** Cube displacement (m) from the same sweep with the hand untracked. */
  pushedWhileUntracked?: number;
  /** Cube rise (m) while a pinch weld holds it, and the hand's own rise. */
  lifted?: number;
  handLifted?: number;
  /** Cube drop (m) in the second after the pinch ends. */
  dropped?: number;
  grabbedDuringLift?: boolean;
  grabbedAfterRelease?: boolean;
}

declare global {
  interface Window {
    __xrHarness?: XrHandOutcome;
  }
}

const MODEL = `<mujoco model="xr-hand-harness">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="cube" pos="0.40 0 0.03">
      <freejoint/>
      <geom name="cubeg" type="box" size=".03 .03 .03" mass=".15"/>
    </body>
  </worldbody>
</mujoco>`;

/** The runtime's default control rate: 0.02 s of 0.002 s substeps. */
const DECIMATION = 10;
const CUBE_HALF = 0.03;

/**
 * A crude right hand, in MuJoCo coordinates: fingertips at the origin, palm and wrist
 * behind it along -x. Enough to touch things with, and to put a pinch midpoint somewhere
 * specific, which is all the physics under test cares about.
 */
const JOINT_OFFSETS: Record<string, [number, number, number]> = {
  wrist: [-0.09, 0, 0],
  [PALM_JOINT]: [-0.05, 0, 0],
  'thumb-phalanx-proximal': [-0.03, -0.02, 0],
  'thumb-tip': [0, -0.012, 0],
  'index-finger-phalanx-proximal': [-0.03, 0.01, 0.01],
  'index-finger-tip': [0, 0.012, 0],
  'middle-finger-phalanx-proximal': [-0.03, 0.02, 0],
  'middle-finger-tip': [-0.005, 0.024, 0],
  'ring-finger-tip': [-0.01, 0.034, 0],
  'pinky-finger-phalanx-proximal': [-0.03, 0.04, 0],
  'pinky-finger-tip': [-0.015, 0.044, 0],
};

function poseHand(
  frames: HandFrames,
  origin: [number, number, number],
  options: { tracked: boolean; pinching: boolean }
): void {
  const frame = frames.right;
  frame.tracked = options.tracked;
  frame.pinching = options.pinching;
  HAND_JOINTS.forEach(({ joint }, i) => {
    const offset = JOINT_OFFSETS[joint] ?? [0, 0, 0];
    const pose: HandJointPose = {
      pos: [origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]],
      quat: [1, 0, 0, 0],
    };
    frame.joints[i] = options.tracked ? pose : null;
  });
}

/** Published only once the run ends, so the spec's wait cannot catch a half-filled result. */
const out: XrHandOutcome = { ok: false };

async function main(): Promise<void> {
  const mujoco = await (await import('mujoco')).default();
  try {
    mujoco.FS.mkdir('/working');
  } catch {
    // already there
  }
  mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
  mujoco.FS.writeFile('/working/harness.xml', MODEL);

  const injected = injectHandRig(MODEL);
  if (!injected) throw new Error('injectHandRig declined the harness model');
  mujoco.FS.writeFile('/working/mjswan-xr-harness.xml', injected);

  const mjModel = mujoco.MjModel.mj_loadXML('/working/mjswan-xr-harness.xml');
  const mjData = new mujoco.MjData(mjModel);
  out.bodies = { total: mjModel.nbody, mocap: mjModel.nmocap, equalities: mjModel.neq };

  const binding = HandMocapBinding.bind(mujoco, mjModel);
  out.bound = binding !== null;
  if (!binding) throw new Error('HandMocapBinding.bind found no rig in the compiled model');

  let cubeId = -1;
  for (let b = 0; b < mjModel.nbody; b++) {
    if (mjModel.body(b).name === 'cube') cubeId = b;
  }
  if (cubeId < 0) throw new Error('cube body missing');
  const graspable = new Set([cubeId]);
  const frames = createHandFrames();
  const cubePos = (axis: number): number => mjData.xpos[cubeId * 3 + axis];
  const palmIndex = HAND_JOINTS.findIndex((entry) => entry.joint === PALM_JOINT);
  const palmZ = (): number => {
    const palm = frames.right.joints[palmIndex];
    return palm ? palm.pos[2] : 0;
  };

  /** One control step, exactly as `executeSimulationSteps` runs it. */
  const step = (): void => {
    binding.beginStep(frames);
    binding.applyGrabs(mjModel, mjData, frames, graspable);
    stepPhysics(
      mujoco,
      mjModel,
      mjData,
      [],
      new Float32Array(0),
      DECIMATION,
      (substep) => binding.writeSubstep(mjModel, mjData, (substep + 1) / DECIMATION)
    );
  };

  const settle = (steps: number): void => {
    for (let i = 0; i < steps; i++) step();
  };

  /** Sweep the hand along +x through the cube's resting place. */
  const sweep = (tracked: boolean): number => {
    mujoco.mj_resetData(mjModel, mjData);
    binding.reset(mjModel, mjData);
    poseHand(frames, [0.15, 0, CUBE_HALF], { tracked, pinching: false });
    mujoco.mj_forward(mjModel, mjData);
    const before = cubePos(0);
    for (let i = 0; i < 60; i++) {
      poseHand(frames, [0.15 + i * 0.005, 0, CUBE_HALF], { tracked, pinching: false });
      step();
    }
    return cubePos(0) - before;
  };

  out.pushed = sweep(true);
  out.pushedWhileUntracked = sweep(false);

  // --- pinch, lift, release ---
  mujoco.mj_resetData(mjModel, mjData);
  binding.reset(mjModel, mjData);
  // Fingertips at the cube, so the pinch midpoint is inside grabbing range.
  poseHand(frames, [0.40, 0, CUBE_HALF], { tracked: true, pinching: false });
  mujoco.mj_forward(mjModel, mjData);
  settle(2);

  poseHand(frames, [0.40, 0, CUBE_HALF], { tracked: true, pinching: true });
  step();
  out.grabbedDuringLift = binding.grabbed('right') === cubeId;

  const cubeBeforeLift = cubePos(2);
  const handBeforeLift = palmZ();
  for (let i = 1; i <= 50; i++) {
    poseHand(frames, [0.40, 0, CUBE_HALF + i * 0.005], { tracked: true, pinching: true });
    step();
  }
  out.lifted = cubePos(2) - cubeBeforeLift;
  out.handLifted = palmZ() - handBeforeLift;

  const cubeBeforeRelease = cubePos(2);
  poseHand(frames, [0.40, 0, CUBE_HALF + 0.25], { tracked: true, pinching: false });
  settle(50);
  out.dropped = cubeBeforeRelease - cubePos(2);
  out.grabbedAfterRelease = binding.grabbed('right') !== null;

  out.ok = true;
}

main()
  .catch((error: unknown) => {
    out.ok = false;
    out.error = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
  })
  .finally(() => {
    window.__xrHarness = out;
  });
