import type { MjModel } from 'mujoco';

/**
 * Whether a body is driven by the solver rather than fixed to the world.
 *
 * True when the body or an ancestor carries a joint. This is the line between a robot's own links
 * and the scenery around them, and it is the same question a ghost has to ask: a policy's body
 * list is its robot, so anything else in the model has no pose in that list to be drawn at.
 *
 * Mocap bodies are deliberately *not* dynamic here. They move, but nothing about a policy's
 * reference describes them -- see the renderer's separate mocap sync, which exists for exactly
 * that reason.
 */
export function isDynamicBody(mjModel: MjModel, bodyId: number): boolean {
  if (bodyId <= 0 || bodyId >= mjModel.nbody) {
    return false;
  }
  let current = bodyId;
  while (current > 0) {
    if (mjModel.body_jntnum[current] > 0) {
      return true;
    }
    current = mjModel.body_parentid[current];
  }
  return false;
}
