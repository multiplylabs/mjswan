import * as THREE from 'three';

export function mjcToThreeCoordinate(v: ArrayLike<number>): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[2], -v[1]);
}

export function threeToMjcCoordinate(v: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3(v.x, -v.z, v.y);
}

/**
 * Three.js (x, y, z, w) → MuJoCo (w, x, y, z): the inverse of the swizzle `getQuaternion`
 * applies (scene.ts). Needed to push a pose *into* MuJoCo — the scene only ever pulled
 * poses out of it.
 */
export function threeToMjcQuaternion(
  q: THREE.Quaternion
): [number, number, number, number] {
  return [-q.w, -q.x, q.z, -q.y];
}
