import type { MjModel } from 'mujoco';

import { bodyIndex, geomIndex } from '../utils/modelNames';
import type {
  CommandConfigEntry,
  CommandTerm,
  CommandTermContext,
  CommandUiConfig,
} from './types';

/**
 * A ramp the operator drops in front of the robot: up, along a flat top, and back down.
 *
 * MuJoCo compiles a model once and a browser cannot add geometry to it afterwards, so the scene
 * carries the three slabs from the start and this term only *moves* them. They are mocap bodies
 * because that is the one kind of geometry a running simulation lets you reposition: static to the
 * solver, so the robot walks on them rather than pushing them, and driven entirely by
 * `mocap_pos`/`mocap_quat`. Their transforms reach the renderer through the ordinary per-frame body
 * sync, which is why the picture follows without touching a mesh.
 *
 * Off, the slabs park far below an opaque ground plane rather than being hidden -- a geom the
 * viewer cannot see but the solver still can would be a trap for whoever debugs a phantom contact.
 *
 * The pitch is redrawn on every placement, from the seeded RNG so a seeded session stays
 * reproducible. Both ramps take the *same* angle: their length is fixed by the compiled model, so
 * an independent descent angle would land the far end above or below the floor and leave a step
 * where the ground should be.
 */

/** Bodies *and* geoms the scene must carry; the term is inert without them. */
const ASCENT = 'slope_ascent';
const PLATEAU = 'slope_plateau';
const DESCENT = 'slope_descent';

const DEFAULT_ANGLE_RANGE_DEG: readonly [number, number] = [10, 15];
/** Metres ahead of the robot the climb begins, leaving room to walk into it. */
const DEFAULT_LEAD_M = 3.0;
/** Well under an opaque floor, and far enough that no camera pull-back brings it into frame. */
const PARK_Z = -80.0;

interface SlopePieces {
  mocap: Record<string, number>;
  halfLength: Record<string, number>;
  halfThickness: number;
}

/** Yaw of a wxyz quaternion, the only component of the robot's orientation a ramp should follow. */
function yawOf(w: number, x: number, y: number, z: number): number {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

export class SlopeTerrain implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly ui: CommandUiConfig | null;
  private readonly angleRangeDeg: readonly [number, number];
  private readonly lead: number;
  private readonly value = new Float32Array(1);

  private pieces: SlopePieces | null = null;
  private resolvedFor: MjModel | null = null;
  private placementPending = false;
  private enabled = false;

  constructor(_termName: string, config: CommandConfigEntry, context: CommandTermContext) {
    this.context = context;
    this.ui = (config.ui as CommandUiConfig | undefined) ?? null;
    const range = config.angle_range_deg as [number, number] | undefined;
    this.angleRangeDeg = range ?? DEFAULT_ANGLE_RANGE_DEG;
    this.lead = (config.lead_m as number | undefined) ?? DEFAULT_LEAD_M;
    // A checkbox declared on by default means the policy this term rides on is the one that can
    // climb, so the ramp is there from the moment it loads rather than waiting to be asked for.
    const declared = this.ui?.inputs?.find((input) => input.type === 'checkbox');
    this.enabled = declared?.default === true;
    this.value[0] = this.enabled ? 1 : 0;
    this.placementPending = this.enabled;
  }

  getUiConfig(): CommandUiConfig | null {
    return this.ui;
  }

  /** The toggle itself: this term drives geometry, not a policy input. */
  getCommand(): Float32Array {
    return this.value;
  }

  getStateField(field: string): Float32Array | null {
    return field === 'command' ? this.value : null;
  }

  getUiValue(): number {
    return this.value[0];
  }

  setValue(_inputName: string, value: number): number {
    const on = value >= 0.5;
    if (on !== this.enabled) {
      this.enabled = on;
      // Placed on the next update, not here: the panel can toggle before a model is loaded, and
      // the robot pose this reads from is only meaningful once the simulation is stepping.
      this.placementPending = on;
      if (!on) this.park();
    }
    this.value[0] = on ? 1 : 0;
    return this.value[0];
  }

  /**
   * Re-place rather than clear, and deliberately keep the toggle: a reset moves the robot back to
   * its start, so a slope left where it was would sit behind it -- or worse, on top of it.
   */
  reset(): void {
    if (this.enabled) this.placementPending = true;
    else this.park();
  }

  update(): void {
    if (this.placementPending && this.place()) this.placementPending = false;
  }

  private resolve(mjModel: MjModel): SlopePieces | null {
    if (this.pieces && this.resolvedFor === mjModel) return this.pieces;
    this.resolvedFor = mjModel;
    this.pieces = null;

    const mocap: Record<string, number> = {};
    const halfLength: Record<string, number> = {};
    let halfThickness = 0;
    for (const name of [ASCENT, PLATEAU, DESCENT]) {
      const body = bodyIndex(mjModel, name);
      const geom = geomIndex(mjModel, name);
      if (body < 0 || geom < 0) return null;
      const id = mjModel.body_mocapid[body];
      if (id < 0) {
        console.warn(`[SlopeTerrain] body "${name}" is not a mocap body; slope disabled.`);
        return null;
      }
      mocap[name] = id;
      // Sizes come off the compiled model rather than being repeated here, so the geometry has
      // exactly one definition -- the scene that built it.
      halfLength[name] = mjModel.geom_size[geom * 3 + 0];
      halfThickness = mjModel.geom_size[geom * 3 + 2];
    }
    this.pieces = { mocap, halfLength, halfThickness };
    return this.pieces;
  }

  private park(): void {
    const { mjModel, mjData } = this.context;
    if (!mjModel || !mjData) return;
    const pieces = this.resolve(mjModel);
    if (!pieces) return;
    for (const name of [ASCENT, PLATEAU, DESCENT]) {
      const m = pieces.mocap[name];
      mjData.mocap_pos[m * 3 + 0] = 0;
      mjData.mocap_pos[m * 3 + 1] = 0;
      mjData.mocap_pos[m * 3 + 2] = PARK_Z;
      mjData.mocap_quat[m * 4 + 0] = 1;
      mjData.mocap_quat[m * 4 + 1] = 0;
      mjData.mocap_quat[m * 4 + 2] = 0;
      mjData.mocap_quat[m * 4 + 3] = 0;
    }
  }

  private place(): boolean {
    const { mjModel, mjData } = this.context;
    if (!mjModel || !mjData || mjModel.nq < 7) return false;
    const pieces = this.resolve(mjModel);
    if (!pieces) return false;

    // The floating base: position and orientation of the robot as it stands right now.
    const yaw = yawOf(mjData.qpos[3], mjData.qpos[4], mjData.qpos[5], mjData.qpos[6]);
    const fx = Math.cos(yaw);
    const fy = Math.sin(yaw);
    const toeX = mjData.qpos[0] + fx * this.lead;
    const toeY = mjData.qpos[1] + fy * this.lead;

    const [lo, hi] = this.angleRangeDeg;
    const pitch = ((this.context.rng?.uniform(lo, hi) ?? lo + Math.random() * (hi - lo)) * Math.PI) / 180;
    const sin = Math.sin(pitch);
    const cos = Math.cos(pitch);

    const a = pieces.halfLength[ASCENT];
    const p = pieces.halfLength[PLATEAU];
    const t = pieces.halfThickness;
    // A slab of surface half-length `a` tilted by `pitch` covers this much ground and climbs this
    // high; the descent mirrors it, which is what puts the far end back on the floor.
    const run = 2 * a * cos;
    const rise = 2 * a * sin;

    // Each slab is placed by its *top* surface -- the face the robot walks on -- then pushed back
    // along its own normal by the half-thickness to get the body centre MuJoCo wants.
    this.put(pieces, ASCENT, toeX, toeY, run / 2 + t * sin, rise / 2 - t * cos, yaw, -pitch);
    this.put(pieces, PLATEAU, toeX, toeY, run + p, rise - t, yaw, 0);
    this.put(pieces, DESCENT, toeX, toeY, 1.5 * run + 2 * p - t * sin, rise / 2 - t * cos, yaw, pitch);
    return true;
  }

  private put(
    pieces: SlopePieces,
    name: string,
    toeX: number,
    toeY: number,
    along: number,
    z: number,
    yaw: number,
    pitch: number
  ): void {
    const mjData = this.context.mjData!;
    const m = pieces.mocap[name];
    mjData.mocap_pos[m * 3 + 0] = toeX + Math.cos(yaw) * along;
    mjData.mocap_pos[m * 3 + 1] = toeY + Math.sin(yaw) * along;
    mjData.mocap_pos[m * 3 + 2] = z;
    // Yaw then pitch, so the ramp climbs along the robot's own forward whatever way it faces.
    const cy = Math.cos(yaw / 2);
    const sy = Math.sin(yaw / 2);
    const cp = Math.cos(pitch / 2);
    const sp = Math.sin(pitch / 2);
    mjData.mocap_quat[m * 4 + 0] = cy * cp;
    mjData.mocap_quat[m * 4 + 1] = -sy * sp;
    mjData.mocap_quat[m * 4 + 2] = cy * sp;
    mjData.mocap_quat[m * 4 + 3] = sy * cp;
  }
}
