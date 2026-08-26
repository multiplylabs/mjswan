import type { MjModel } from 'mujoco';

/**
 * The names of one entity kind, in index order.
 *
 * MuJoCo keeps every name in a single NUL-separated block and addresses them per entity kind, so
 * reading one back means decoding from its offset to the next NUL rather than indexing a table.
 */
export function decodeNames(
  mjModel: MjModel,
  count: number,
  adr: ArrayLike<number>
): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let i = 0; i < count; i++) {
    const start = adr[i];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

/** Index of a named body, or -1. */
export function bodyIndex(mjModel: MjModel, name: string): number {
  return decodeNames(mjModel, mjModel.nbody, mjModel.name_bodyadr).indexOf(name);
}

/** Index of a named geom, or -1. */
export function geomIndex(mjModel: MjModel, name: string): number {
  return decodeNames(mjModel, mjModel.ngeom, mjModel.name_geomadr).indexOf(name);
}
