import { describe, expect, it } from 'vitest';

import { SlopeTerrain } from '../slopeTerrain';
import type { CommandTermContext } from '../types';

const NAMES = ['world', 'pelvis', 'slope_ascent', 'slope_plateau', 'slope_descent'];

/** A model carrying just the fields the term reads: names, mocap ids and geom sizes. */
function stubModel() {
  const encoder = new TextEncoder();
  const bytes: number[] = [];
  const adr: number[] = [];
  for (const name of NAMES) {
    adr.push(bytes.length);
    bytes.push(...encoder.encode(name), 0);
  }
  // Geoms share the body names, minus `world`/`pelvis` which carry none the term looks up.
  const geomAdr = [adr[2], adr[3], adr[4]];
  return {
    names: new Uint8Array(bytes).buffer,
    nbody: NAMES.length,
    ngeom: 3,
    nq: 7,
    name_bodyadr: adr,
    name_geomadr: geomAdr,
    // world and pelvis are not mocap; the three slabs are mocap 0, 1, 2.
    body_mocapid: [-1, -1, 0, 1, 2],
    geom_size: [1.25, 1.0, 0.06, 1.0, 1.0, 0.06, 1.25, 1.0, 0.06],
  } as never;
}

function stubData(x: number, y: number, yaw: number) {
  return {
    qpos: [x, y, 0.793, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)],
    mocap_pos: new Float64Array(9),
    mocap_quat: new Float64Array(12),
  } as never;
}

function makeTerm(data: unknown, angleDeg: number) {
  const context = {
    mjModel: stubModel(),
    mjData: data,
    scene: {},
    // A fixed draw, so the placement is checkable.
    rng: { uniform: () => angleDeg },
  } as unknown as CommandTermContext;
  const term = new SlopeTerrain(
    'terrain',
    { name: 'SlopeTerrain', angle_range_deg: [angleDeg, angleDeg], lead_m: 3.0 },
    context
  );
  return term;
}

describe('SlopeTerrain', () => {
  it('parks the slabs below the floor until switched on', () => {
    const data = stubData(0, 0, 0) as unknown as { mocap_pos: Float64Array };
    const term = makeTerm(data, 12);
    term.update!();
    expect(Array.from(data.mocap_pos)).toEqual(new Array(9).fill(0));
  });

  it('places a continuous ascent, top and descent ahead of the robot', () => {
    const angle = 12;
    const data = stubData(0, 0, 0) as unknown as {
      mocap_pos: Float64Array;
      mocap_quat: Float64Array;
    };
    const term = makeTerm(data, angle);
    term.setValue!('enabled', 1);
    term.update!();

    const rad = (angle * Math.PI) / 180;
    const a = 1.25;
    const t = 0.06;
    const run = 2 * a * Math.cos(rad);
    const rise = 2 * a * Math.sin(rad);

    // Ascent: its top face starts on the floor 3 m ahead and climbs.
    expect(data.mocap_pos[0]).toBeCloseTo(3 + run / 2 + t * Math.sin(rad), 6);
    expect(data.mocap_pos[2]).toBeCloseTo(rise / 2 - t * Math.cos(rad), 6);
    // Plateau: flat, one thickness under the height the ascent reached.
    expect(data.mocap_pos[3]).toBeCloseTo(3 + run + 1.0, 6);
    expect(data.mocap_pos[5]).toBeCloseTo(rise - t, 6);
    // Descent mirrors the ascent, so its far end returns to the floor.
    expect(data.mocap_pos[8]).toBeCloseTo(rise / 2 - t * Math.cos(rad), 6);
    // Pitched about the lateral axis only: no yaw component for a robot facing +x.
    expect(data.mocap_quat[1]).toBeCloseTo(0, 6);
    expect(data.mocap_quat[2]).toBeCloseTo(Math.sin(-rad / 2), 6);
  });

  it('places along the robot heading, not the world axis', () => {
    const data = stubData(1, 2, Math.PI / 2) as unknown as { mocap_pos: Float64Array };
    const term = makeTerm(data, 12);
    term.setValue!('enabled', 1);
    term.update!();
    // Facing +y: the ramp advances in y and holds the robot's x.
    expect(data.mocap_pos[0]).toBeCloseTo(1, 6);
    expect(data.mocap_pos[1]).toBeGreaterThan(2 + 3);
  });

  it('parks again when switched off', () => {
    const data = stubData(0, 0, 0) as unknown as { mocap_pos: Float64Array };
    const term = makeTerm(data, 12);
    term.setValue!('enabled', 1);
    term.update!();
    term.setValue!('enabled', 0);
    expect(data.mocap_pos[2]).toBeLessThan(-10);
  });
});
