/**
 * The Three.js side of hand tracking: owns the two `XRHandSpace`s, renders them, and
 * snapshots their joint poses into MuJoCo coordinates once per XR frame.
 *
 * The snapshot exists because the two loops run on different clocks — WebXR joint poses
 * are only valid inside the XR animation frame (72–90 Hz on a Quest), while the step loop
 * consumes them at the control rate (50 Hz by default). The render loop publishes here;
 * `HandMocapBinding` reads.
 */

import * as THREE from 'three';
import type { XRHandJoints } from 'three';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';

import { threeToMjcCoordinate, threeToMjcQuaternion } from '../scene/coordinate';
import { HAND_JOINTS, type Handedness } from './handRig';
import { createHandFrames, type HandFrames, type HandJointPose } from './HandMocap';

/** WebXR reports at most two hands, and three.js keys them by input-source index. */
const HAND_INDICES = [0, 1] as const;

interface TrackedHand {
  space: THREE.XRHandSpace;
  /** Known only once the input source connects; 'none' handedness stays unbound. */
  handedness: Handedness | null;
  pinching: boolean;
  listeners: { type: string; handler: (event: { data?: XRInputSource }) => void }[];
}

export class XrHandTracker {
  /** Latest published poses, in MuJoCo coordinates. Overwritten in place each frame. */
  readonly frames: HandFrames = createHandFrames();

  private readonly hands: TrackedHand[] = [];
  /** Reused so a 72 Hz snapshot of 22 joints allocates nothing per frame. */
  private readonly pool: Record<Handedness, HandJointPose[]>;
  private readonly tmpPos = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();
  private readonly tmpScale = new THREE.Vector3();

  constructor(
    renderer: THREE.WebGLRenderer,
    private readonly rig: THREE.Object3D
  ) {
    const pose = (): HandJointPose => ({ pos: [0, 0, 0], quat: [1, 0, 0, 0] });
    this.pool = {
      left: HAND_JOINTS.map(pose),
      right: HAND_JOINTS.map(pose),
    };

    const factory = new XRHandModelFactory();
    for (const index of HAND_INDICES) {
      const space = renderer.xr.getHand(index);
      const hand: TrackedHand = { space, handedness: null, pinching: false, listeners: [] };

      const bind = (type: string, handler: (event: { data?: XRInputSource }) => void): void => {
        space.addEventListener(type as never, handler as never);
        hand.listeners.push({ type, handler });
      };
      bind('connected', (event) => {
        const handedness = event.data?.handedness;
        hand.handedness = handedness === 'left' || handedness === 'right' ? handedness : null;
      });
      bind('disconnected', () => {
        hand.handedness = null;
        hand.pinching = false;
      });
      bind('pinchstart', () => {
        hand.pinching = true;
      });
      bind('pinchend', () => {
        hand.pinching = false;
      });

      // Spheres, not the mesh profile: the mesh models are fetched from a CDN, and a
      // built mjswan app is a self-contained static site.
      if (!space.userData.mjswanHandModel) {
        space.userData.mjswanHandModel = true;
        space.add(factory.createHandModel(space, 'spheres'));
      }
      rig.add(space);
      this.hands.push(hand);
    }
  }

  /** Republish `frames` from this frame's joint poses. Call from the XR animation loop. */
  snapshot(): void {
    for (const frame of Object.values(this.frames)) {
      frame.tracked = false;
      frame.pinching = false;
      frame.joints.fill(null);
    }

    for (const hand of this.hands) {
      const handedness = hand.handedness;
      if (!handedness || !hand.space.visible) continue;
      const frame = this.frames[handedness];
      const pool = this.pool[handedness];
      // The rig's transform is part of the answer, so resolve the whole chain.
      hand.space.updateWorldMatrix(true, true);

      let tracked = false;
      for (let i = 0; i < HAND_JOINTS.length; i++) {
        const joint = hand.space.joints[HAND_JOINTS[i].joint as keyof XRHandJoints];
        if (!joint || !joint.visible) continue;
        joint.matrixWorld.decompose(this.tmpPos, this.tmpQuat, this.tmpScale);
        const pos = threeToMjcCoordinate(this.tmpPos);
        const quat = threeToMjcQuaternion(this.tmpQuat);
        const pose = pool[i];
        pose.pos[0] = pos.x;
        pose.pos[1] = pos.y;
        pose.pos[2] = pos.z;
        pose.quat[0] = quat[0];
        pose.quat[1] = quat[1];
        pose.quat[2] = quat[2];
        pose.quat[3] = quat[3];
        frame.joints[i] = pose;
        tracked = true;
      }
      frame.tracked = tracked;
      frame.pinching = tracked && hand.pinching;
    }
  }

  dispose(): void {
    for (const hand of this.hands) {
      for (const { type, handler } of hand.listeners) {
        hand.space.removeEventListener(type as never, handler as never);
      }
      hand.listeners.length = 0;
      this.rig.remove(hand.space);
    }
    this.hands.length = 0;
  }
}
