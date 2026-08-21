/**
 * The MJCF the viewer injects so a tracked hand can exist inside the physics: one mocap
 * body per tracked joint, plus one inactive weld per hand for pinch grabs.
 *
 * Mocap bodies are what make the hand a *body* rather than a force: they carry no dofs,
 * so `mocap_pos`/`mocap_quat` teleport them each step while MuJoCo still resolves their
 * contacts. Carrying no dofs is also why injection is safe for a policy — `nq`/`nv` are
 * untouched, and the new bodies/geoms/equalities are appended, so no existing id shifts.
 */

export type Handedness = 'left' | 'right';

export const HANDEDNESS: readonly Handedness[] = ['left', 'right'];

/** Prefix on every injected name, so a re-injection is detectable and ids are findable. */
export const XR_PREFIX = 'mjswan_xr_';

/**
 * The tracked subset of WebXR's 25 hand joints: five fingertips to touch with, four
 * proximal knuckles and the wrist/palm to push with. Radii are nominal adult-hand values
 * — the XR runtime's own `jointRadius` arrives only after a session starts, long after
 * the model is compiled.
 */
export const HAND_JOINTS: readonly { readonly joint: string; readonly radius: number }[] = [
  { joint: 'wrist', radius: 0.024 },
  { joint: 'middle-finger-metacarpal', radius: 0.026 },
  { joint: 'thumb-phalanx-proximal', radius: 0.013 },
  { joint: 'thumb-tip', radius: 0.011 },
  { joint: 'index-finger-phalanx-proximal', radius: 0.012 },
  { joint: 'index-finger-tip', radius: 0.009 },
  { joint: 'middle-finger-phalanx-proximal', radius: 0.012 },
  { joint: 'middle-finger-tip', radius: 0.009 },
  { joint: 'ring-finger-tip', radius: 0.009 },
  { joint: 'pinky-finger-phalanx-proximal', radius: 0.011 },
  { joint: 'pinky-finger-tip', radius: 0.008 },
];

/** Palm proxy: the weld's body1, and the frame a grabbed body is held in. */
export const PALM_JOINT = 'middle-finger-metacarpal';

/** Pinch endpoints, in the order `pinchMidpoint` expects them. */
export const PINCH_JOINTS = ['thumb-tip', 'index-finger-tip'] as const;

export function handBodyName(hand: Handedness, joint: string): string {
  return `${XR_PREFIX}${hand}_${joint}`;
}

export function handWeldName(hand: Handedness): string {
  return `${XR_PREFIX}grab_${hand}`;
}

/**
 * Parked a metre up rather than at the origin: an untracked hand should sit clear of the
 * scene even in the frame before `setCollisionsEnabled` catches up.
 */
export const PARK_POS: readonly [number, number, number] = [0, 0, 1];

/**
 * Soft-ish contacts (default `solref` is `0.02 1`) because a body that teleports rather
 * than accelerates arrives with no velocity for the solver to work from: a stiffer
 * contact turns a one-step overlap into a launch. `condim="4"` adds the torsional
 * friction that keeps a pinched object from spinning off a fingertip.
 */
function geomXml(radius: number): string {
  return (
    `<geom type="sphere" size="${radius}" group="3" condim="4" ` +
    `friction="1.2 0.02 0.001" solref="0.015 1" solimp="0.9 0.95 0.001"/>`
  );
}

function handXml(hand: Handedness): string {
  return HAND_JOINTS.map(
    ({ joint, radius }) =>
      `    <body name="${handBodyName(hand, joint)}" mocap="true" pos="${PARK_POS.join(' ')}">\n` +
      `      ${geomXml(radius)}\n` +
      `    </body>`
  ).join('\n');
}

/** The fragment `injectHandRig` splices in; separate sections merge into the model's own. */
function handRigXml(): string {
  const worldbody = HANDEDNESS.map(handXml).join('\n');
  const welds = HANDEDNESS.map(
    (hand) =>
      `    <weld name="${handWeldName(hand)}" body1="${handBodyName(hand, PALM_JOINT)}" ` +
      `body2="world" active="false" torquescale="1"/>`
  ).join('\n');
  return (
    `\n  <!-- mjswan: injected XR hand rig -->\n` +
    `  <worldbody>\n${worldbody}\n  </worldbody>\n` +
    `  <equality>\n${welds}\n  </equality>\n`
  );
}

/**
 * Splice the rig into a model's root XML, or return null when there is nothing to splice
 * into (an already-injected model, or a root element the text edit cannot reach).
 * Repeated `<worldbody>`/`<equality>` sections are legal MJCF and merge on compile.
 */
export function injectHandRig(xml: string): string | null {
  if (xml.includes(XR_PREFIX)) {
    return null;
  }
  const close = xml.lastIndexOf('</mujoco>');
  if (close < 0) {
    return null;
  }
  return xml.slice(0, close) + handRigXml() + xml.slice(close);
}
