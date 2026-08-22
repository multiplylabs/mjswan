/**
 * Applies an operator-controlled external wrench to named bodies, from a UI command's values.
 *
 * The use case is perturbation testing: a slider per axis that pushes on a specific body, so a
 * policy trained against external disturbances can be exercised deliberately rather than only by
 * mouse-dragging. Distinct from the viewer's drag forces, which are transient and follow the
 * pointer.
 *
 * Values are read by input **name**, not by offset into the command vector, so reordering or
 * adding a UI input cannot silently repoint an axis at the wrong slider.
 *
 * Forces are in **world** coordinates and are applied at the body origin, which is what
 * `mjData.xfrc_applied` takes. Targets are written (not accumulated) every step, so a disabled
 * target reliably returns to zero whether or not anything else clears the buffer.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** One body the operator can push, and the UI inputs that say how hard. */
export interface ExternalWrenchTarget {
  /** Body name in the compiled model; an entity-prefixed model is matched by suffix too. */
  body: string;
  /** UI input names supplying world-frame force x, y, z, in newtons. */
  axes: [string, string, string];
  /** Optional checkbox input gating the whole target. Absent means always on. */
  enable?: string;
  /** Optional UI input names supplying a world-frame torque x, y, z, in newton-metres. */
  torque_axes?: [string, string, string];
}

export interface ExternalWrenchConfig {
  /** Command term whose UI values drive the wrench. */
  command_name: string;
  targets: ExternalWrenchTarget[];
}

/** A term exposing its UI values by input name, as `UiCommand` and `OnnxCommand` both do. */
interface UiValueSource {
  getUiValue?(inputName: string): number | undefined;
}

export function isExternalWrenchConfig(value: unknown): value is ExternalWrenchConfig {
  if (!value || typeof value !== 'object') return false;
  const cfg = value as Partial<ExternalWrenchConfig>;
  return typeof cfg.command_name === 'string' && Array.isArray(cfg.targets);
}

function decodeBodyNames(mjModel: MjModel): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let b = 0; b < mjModel.nbody; b++) {
    const start = mjModel.name_bodyadr[b];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

/**
 * Resolves target body ids once per model, then writes each step.
 *
 * A target naming a body the model does not have is dropped with one warning rather than
 * throwing, so a config outliving a scene swap degrades instead of breaking the run.
 */
export class ExternalWrenchApplier {
  private readonly bodyIds: (number | null)[];

  constructor(
    private readonly config: ExternalWrenchConfig,
    mjModel: MjModel,
  ) {
    const names = decodeBodyNames(mjModel);
    this.bodyIds = config.targets.map((target) => {
      let id = names.indexOf(target.body);
      if (id < 0) {
        // mjlab's `attach` prefixes names with `entity/`; match the bare name too.
        id = names.findIndex((name) => name.endsWith(`/${target.body}`));
      }
      if (id <= 0) {
        console.warn(`[ExternalWrench] no body named '${target.body}' in this model`);
        return null;
      }
      return id;
    });
  }

  /** Write this step's wrench for every resolved target. */
  apply(mjData: MjData, term: UiValueSource | undefined): void {
    if (!term?.getUiValue) return;
    const read = (name: string | undefined): number =>
      name === undefined ? 0 : (term.getUiValue?.(name) ?? 0);

    for (let t = 0; t < this.config.targets.length; t++) {
      const bodyId = this.bodyIds[t];
      if (bodyId === null) continue;
      const target = this.config.targets[t];
      const gate = target.enable === undefined ? 1 : read(target.enable) >= 0.5 ? 1 : 0;
      const base = bodyId * 6;
      for (let axis = 0; axis < 3; axis++) {
        mjData.xfrc_applied[base + axis] = gate * read(target.axes[axis]);
        mjData.xfrc_applied[base + 3 + axis] = gate * read(target.torque_axes?.[axis]);
      }
    }
  }
}
