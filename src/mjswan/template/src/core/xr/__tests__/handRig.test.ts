import { describe, expect, it } from 'vitest';

import { createHandFrames } from '../HandMocap';
import {
  HAND_JOINTS,
  HANDEDNESS,
  PALM_JOINT,
  PINCH_JOINTS,
  handBodyName,
  handWeldName,
  injectHandRig,
} from '../handRig';

/** The WebXR hand-input joint names, verbatim from the spec's `XRHandJoint` enum. */
const XR_HAND_JOINTS = new Set([
  'wrist',
  'thumb-metacarpal',
  'thumb-phalanx-proximal',
  'thumb-phalanx-distal',
  'thumb-tip',
  'index-finger-metacarpal',
  'index-finger-phalanx-proximal',
  'index-finger-phalanx-intermediate',
  'index-finger-phalanx-distal',
  'index-finger-tip',
  'middle-finger-metacarpal',
  'middle-finger-phalanx-proximal',
  'middle-finger-phalanx-intermediate',
  'middle-finger-phalanx-distal',
  'middle-finger-tip',
  'ring-finger-metacarpal',
  'ring-finger-phalanx-proximal',
  'ring-finger-phalanx-intermediate',
  'ring-finger-phalanx-distal',
  'ring-finger-tip',
  'pinky-finger-metacarpal',
  'pinky-finger-phalanx-proximal',
  'pinky-finger-phalanx-intermediate',
  'pinky-finger-phalanx-distal',
  'pinky-finger-tip',
]);

const MODEL = `<mujoco model="m">
  <worldbody>
    <body name="cube"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
</mujoco>
`;

describe('handRig', () => {
  /** A joint the XR runtime never reports is a mocap body that never moves. */
  it('tracks only real WebXR joints', () => {
    for (const { joint } of HAND_JOINTS) {
      expect(XR_HAND_JOINTS.has(joint), joint).toBe(true);
    }
    const tracked = new Set(HAND_JOINTS.map((entry) => entry.joint));
    expect(tracked.size).toBe(HAND_JOINTS.length);
    expect(tracked.has(PALM_JOINT)).toBe(true);
    for (const joint of PINCH_JOINTS) {
      expect(tracked.has(joint), joint).toBe(true);
    }
  });

  it('splices a mocap body per joint and one weld per hand into the model', () => {
    const injected = injectHandRig(MODEL);
    expect(injected).not.toBeNull();
    const xml = injected as string;

    for (const hand of HANDEDNESS) {
      for (const { joint } of HAND_JOINTS) {
        expect(xml).toContain(`name="${handBodyName(hand, joint)}" mocap="true"`);
      }
      expect(xml).toContain(`name="${handWeldName(hand)}"`);
    }
    expect(xml.match(/mocap="true"/g)).toHaveLength(HANDEDNESS.length * HAND_JOINTS.length);
    expect(xml.match(/<weld /g)).toHaveLength(HANDEDNESS.length);
    // Spliced inside the root, and the model's own content is untouched.
    expect(xml.trimEnd().endsWith('</mujoco>')).toBe(true);
    expect(xml).toContain('<body name="cube">');
    // Every injected geom is invisible to the Three.js scene builder (geom_group < 3).
    expect(xml.match(/group="3"/g)).toHaveLength(HANDEDNESS.length * HAND_JOINTS.length);
  });

  it('declines a model it has already been spliced into', () => {
    const once = injectHandRig(MODEL) as string;
    expect(injectHandRig(once)).toBeNull();
  });

  it('declines a model with no closing root tag', () => {
    expect(injectHandRig('<mujoco/>')).toBeNull();
  });

  it('publishes one frame slot per tracked joint', () => {
    const frames = createHandFrames();
    for (const hand of HANDEDNESS) {
      expect(frames[hand].joints).toHaveLength(HAND_JOINTS.length);
      expect(frames[hand].tracked).toBe(false);
    }
  });
});
