import type { MainModule, MjData, MjModel } from 'mujoco';
import type { Scene } from 'three';

import type { CommandsConfig } from '../command';
import type { CommandManager } from '../command/CommandManager';
import type { Bytes } from '../utils/bytes';

export type PolicyRunnerContext = {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene?: Scene | null;
  /** Instance-scoped command manager; command-state input slots read it via the runner. */
  commandManager?: CommandManager;
};

export type PolicyState = {
  jointPos: Float32Array;
  jointVel?: Float32Array;
  rootPos?: Float32Array;
  rootQuat?: Float32Array;
  rootLinVel?: Float32Array;
  rootAngVel?: Float32Array;
  [key: string]: unknown;
};

export type ObservationConfigEntry = {
  name: string;
  [key: string]: unknown;
};

export type ObservationGroupConfig =
  | ObservationConfigEntry[]
  | {
    history_steps?: number;
    interleaved?: boolean;
    components?: ObservationConfigEntry[];
  };

export type ActionConfigEntry = {
  type: string;
  scale?: number | number[] | Record<string, number>;
  offset?: number | Record<string, number>;
  use_default_offset?: boolean;
  stiffness?: number | number[] | Record<string, number>;
  damping?: number | number[] | Record<string, number>;
  actuator_names?: string[];
  [key: string]: unknown;
};

export type TerminationConfigEntry = {
  name: string;
  params?: Record<string, unknown>;
  time_out?: boolean;
};

export type PolicyConfig = {
  policy_module?: string;
  policy_joint_names?: string[];
  policy_num_actions?: number;
  default_joint_pos?: number[];
  encoder_bias?: number[];
  action_scale?: number[] | number;
  stiffness?: number[] | number;
  damping?: number[] | number;
  control_type?: string;
  /**
   * Symmetric bound on the raw policy output, mirroring rsl-rl's
   * `RslRlVecEnvWrapper`. It clamps before `env.step`, so the clamped vector is what
   * the action terms and any `last_action` observation see — not `ActionConfigEntry.clip`,
   * which bounds `raw * scale + offset` per target.
   */
  clip_actions?: number;
  /**
   * Per-input tensor shapes, keyed by ONNX input name, for a graph whose inputs are not flat
   * `[1, N]` vectors. An observation group still produces a flat buffer; this is how the runtime
   * knows to hand it over as, say, `[1, 8, 33, 4]`. Absent (or inconsistent with the buffer's
   * length) falls back to `[1, N]`.
   */
  policy_input_shapes?: Record<string, number[]>;
  /**
   * Value the stored-action buffer holds before the first inference, defaulting to zeros. A
   * policy whose output is an absolute joint target (not a residual) sets its default pose, so
   * the action terms and the `prev_action` slot do not start from a whole-pose error.
   */
  initial_action?: number[];
  /**
   * Bodies an operator can push with a UI command's sliders, for perturbation testing. See
   * `core/engine/externalWrench.ts` for the shape and semantics.
   */
  external_wrench?: {
    command_name: string;
    targets: Array<{
      body: string;
      axes: [string, string, string];
      enable?: string;
      torque_axes?: [string, string, string];
    }>;
  };
  /**
   * The virtual Kelvin-Voigt contact a force-exertion policy pushes against, and the source of the
   * exerted-force reading. See `core/engine/handSpringContact.ts`.
   */
  hand_spring?: {
    command_name: string;
    anchor_body: string;
    targets: Array<{ body: string; hand: number }>;
    max_lead: number;
    smooth_beta?: number;
    two_sided?: boolean;
    damping?: boolean;
    dt: number;
  };
  onnx?: {
    // Weights arrive as bytes via PolicyInput.onnx; policy.json holds only the io keys.
    meta?: {
      in_keys?: string[];
      out_keys?: (string | string[])[];
    };
  };
  commands?: CommandsConfig;
  motions?: Array<{
    name: string;
    /** Injected by the engine from PolicyInput.motions (matched by name). */
    data?: Bytes;
    anchor_body_name: string;
    body_names: string[];
    dataset_joint_names?: string[];
    default?: boolean;
    [key: string]: unknown;
  }>;
  observations?: Record<string, ObservationGroupConfig>;
  actions?: Record<string, ActionConfigEntry>;
  terminations?: Record<string, TerminationConfigEntry>;
  [key: string]: unknown;
};
