/**
 * A translucent tinted copy of the robot, posed from world body transforms.
 *
 * `TrackingCommand`'s reference ghost derives its pose by writing a clip frame into a spare
 * `MjData` and running `mj_forward`. This one is for the other case: a term that already *has*
 * world body poses — a traced graph publishing a target pose, say — and only needs them drawn. So
 * it takes `[nbody, 3]` positions and `[nbody, 4]` quaternions and places the clones directly.
 *
 * Poses arrive in **MuJoCo world coordinates**, with quaternions **xyzw** — the convention a policy
 * contract uses. The MuJoCo-to-three axis swizzle is not applied here: it is `getPosition` /
 * `getQuaternion`'s job, and having one place that knows it is what keeps a second ghost from
 * quietly disagreeing with the first.
 */

import * as THREE from 'three';

import { getPosition, getQuaternion } from './scene';

type MjModel = import('mujoco').MjModel;

export interface PoseGhostOptions {
  /** Shown in the three.js tree; useful when more than one ghost is in a scene. */
  name: string;
  /** Tint, linear RGB in [0, 1]. */
  color: [number, number, number];
  opacity: number;
}

function tint(material: THREE.Material, options: PoseGhostOptions): THREE.Material {
  const next = material.clone();
  if ('transparent' in next) next.transparent = true;
  if ('opacity' in next) next.opacity = options.opacity;
  // Off, so the ghost never occludes the robot it is being compared against.
  if ('depthWrite' in next) next.depthWrite = false;
  if ('color' in next && next.color instanceof THREE.Color) {
    next.color = new THREE.Color(...options.color);
  }
  return next;
}

function hasMesh(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) found = true;
  });
  return found;
}

export class PoseGhost {
  /** Clone per model body id; a body with nothing to draw is absent rather than empty. */
  private readonly bodies = new Map<number, THREE.Group>();
  private readonly root: THREE.Group;
  /** xyzw -> wxyz staging, reused so posing allocates nothing per frame. */
  private readonly scratchQuat = new Float32Array(4);

  constructor(
    parent: THREE.Object3D,
    sourceBodies: Record<number, THREE.Group>,
    mjModel: MjModel,
    private readonly options: PoseGhostOptions,
  ) {
    this.root = new THREE.Group();
    this.root.name = options.name;
    this.root.visible = false;
    for (const [key, body] of Object.entries(sourceBodies)) {
      const bodyId = Number(key);
      if (bodyId <= 0 || bodyId >= mjModel.nbody) continue;
      const clone = body.clone(true);
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.material = Array.isArray(obj.material)
            ? obj.material.map((m) => tint(m, options))
            : tint(obj.material, options);
          // Above the robot and the reference ghost, so a coincident pose still reads.
          obj.renderOrder = 3;
          obj.castShadow = false;
        }
      });
      if (!hasMesh(clone)) continue;
      this.bodies.set(bodyId, clone);
      this.root.add(clone);
    }
    parent.add(this.root);
  }

  /**
   * Place the clones. `pos`/`quat` are flat and indexed by **body order excluding the worldbody**,
   * which is how a policy contract lists bodies — hence `bodyId - 1`.
   */
  update(visible: boolean, pos: Float32Array | null, quat: Float32Array | null): void {
    if (!visible || !pos || !quat) {
      this.root.visible = false;
      return;
    }
    for (const [bodyId, body] of this.bodies) {
      const i = bodyId - 1;
      if ((i + 1) * 3 > pos.length || (i + 1) * 4 > quat.length) continue;
      getPosition(pos, i, body.position);
      // `getQuaternion` reads wxyz, as every MuJoCo buffer in the engine is.
      this.scratchQuat[0] = quat[i * 4 + 3];
      this.scratchQuat[1] = quat[i * 4 + 0];
      this.scratchQuat[2] = quat[i * 4 + 1];
      this.scratchQuat[3] = quat[i * 4 + 2];
      getQuaternion(this.scratchQuat, 0, body.quaternion);
    }
    this.root.visible = true;
  }

  dispose(): void {
    this.root.parent?.remove(this.root);
    this.root.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        if (Array.isArray(obj.material)) for (const m of obj.material) m.dispose?.();
        else obj.material?.dispose?.();
      }
    });
  }
}
