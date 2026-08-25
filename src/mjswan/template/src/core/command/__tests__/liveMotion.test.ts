import { describe, expect, it } from 'vitest';

import { parseFrameBlock } from '../liveMotion';
import fixture from './liveMotion.fixture.json';

/**
 * The fixture is bytes produced by the Python server's own packer, not by a hand-written encoder.
 * A frame layout agreed on in two languages is exactly the kind of thing that goes wrong silently
 * -- a field in the wrong order still parses, and the robot merely tracks badly -- so the test
 * checks against the real thing.
 */
function fixtureBytes(): ArrayBuffer {
  const binary = atob(fixture.base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

describe('parseFrameBlock', () => {
  it('reads the header the server wrote', () => {
    const { start, count } = parseFrameBlock(fixtureBytes(), fixture.n_dofs, fixture.n_bodies);
    expect(start).toBe(fixture.start);
    expect(count).toBe(fixture.count);
  });

  it('recovers every field, frame for frame', () => {
    const { frames } = parseFrameBlock(fixtureBytes(), fixture.n_dofs, fixture.n_bodies);
    const fields = {
      jointPos: fixture.expect.joint_pos,
      jointVel: fixture.expect.joint_vel,
      bodyPosW: fixture.expect.body_pos_w,
      bodyQuatW: fixture.expect.body_quat_w,
      bodyLinVelW: fixture.expect.body_lin_vel_w,
      bodyAngVelW: fixture.expect.body_ang_vel_w,
    } as const;
    for (const [name, expected] of Object.entries(fields)) {
      const got = frames[name as keyof typeof frames];
      expect(got, name).toHaveLength(fixture.count);
      for (let i = 0; i < fixture.count; i++) {
        expect(Array.from(got[i]!), `${name}[${i}]`).toEqual(expected[i]);
      }
    }
  });

  it('copies frames rather than viewing the received buffer', () => {
    const buffer = fixtureBytes();
    const { frames } = parseFrameBlock(buffer, fixture.n_dofs, fixture.n_bodies);
    // A view would keep the whole block alive for as long as one frame is in the ring, and would
    // also alias the next block if the socket reused the buffer.
    expect(frames.jointPos[0]!.buffer).not.toBe(buffer);
    expect(frames.jointPos[0]!.byteOffset).toBe(0);
  });
});
