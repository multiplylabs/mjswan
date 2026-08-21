import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  mjcToThreeCoordinate,
  threeToMjcCoordinate,
  threeToMjcQuaternion,
} from '../coordinate';
import { getQuaternion } from '../scene';
import { SeededRng } from '../../rng';

/** The scene's swizzle, reached through the function the runtime actually calls. */
function mjcToThreeQuaternion(mjc: readonly number[]): THREE.Quaternion {
  const buffer = new Float32Array(mjc);
  return getQuaternion(buffer, 0, new THREE.Quaternion());
}

function randomMjcQuat(rng: SeededRng): number[] {
  const q = new THREE.Quaternion(
    rng.uniform(-1, 1),
    rng.uniform(-1, 1),
    rng.uniform(-1, 1),
    rng.uniform(-1, 1)
  ).normalize();
  return [q.w, q.x, q.y, q.z];
}

describe('coordinate', () => {
  it('round-trips positions', () => {
    const rng = new SeededRng(7);
    for (let i = 0; i < 64; i++) {
      const mjc = [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)];
      const back = threeToMjcCoordinate(mjcToThreeCoordinate(mjc));
      expect([back.x, back.y, back.z]).toEqual(mjc);
    }
  });

  it('round-trips quaternions through the scene swizzle', () => {
    const rng = new SeededRng(11);
    for (let i = 0; i < 256; i++) {
      const mjc = randomMjcQuat(rng);
      const back = threeToMjcQuaternion(mjcToThreeQuaternion(mjc));
      // q and -q are the same rotation; the swizzle pair negates.
      const sign = Math.sign(back[0] * mjc[0]) || 1;
      for (let k = 0; k < 4; k++) expect(back[k] * sign).toBeCloseTo(mjc[k], 6);
    }
  });

  /**
   * The two swizzles describe *one* change of basis, so rotating in MuJoCo and converting
   * must equal converting and rotating in Three.js. Without this, a hand pose written to
   * `mocap_quat` would be silently mirrored.
   */
  it('agrees with the position swizzle as a frame change', () => {
    const rng = new SeededRng(13);
    for (let i = 0; i < 256; i++) {
      const mjc = randomMjcQuat(rng);
      const v = [rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)];
      const rotatedInMjc = new THREE.Vector3(v[0], v[1], v[2]).applyQuaternion(
        new THREE.Quaternion(mjc[1], mjc[2], mjc[3], mjc[0])
      );
      const lhs = mjcToThreeCoordinate([rotatedInMjc.x, rotatedInMjc.y, rotatedInMjc.z]);
      const rhs = mjcToThreeCoordinate(v).applyQuaternion(mjcToThreeQuaternion(mjc));
      expect(lhs.distanceTo(rhs)).toBeLessThan(1e-6);
    }
  });
});
