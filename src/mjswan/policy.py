"""Policy configuration and management.

This module defines the PolicyConfig dataclass and PolicyHandle class for
ONNX policy configuration and command management.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import onnx

from .command import CommandTermConfig
from .motion import MotionConfig, MotionHandle

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
    from .scene import SceneHandle


@dataclass
class PolicyConfig:
    """Configuration for an ONNX policy."""

    name: str
    """Name of the policy."""

    model: onnx.ModelProto
    """ONNX model for the policy."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the policy."""

    source_path: str | None = None
    """Optional source path for the policy ONNX file."""

    config_path: str | None = None
    """Optional source path for the policy config JSON file."""

    commands: dict[str, CommandTermConfig] = field(default_factory=dict)
    """Command terms keyed by their policy-visible names."""

    observations: dict[str, ObservationGroupCfg] | None = None
    """Observation group configurations (mjlab-compatible).

    Keys are **ONNX input names**, not free-form labels: the runtime feeds each
    group's vector as the input of that name, and ``in_keys`` defaults to
    ``["policy"]``. mjlab calls the same group ``"actor"``, so
    :func:`mjswan.adapters.adapt_observations` renames a single group to
    ``"policy"`` on the way in. Values are ``ObservationGroupCfg`` instances whose
    terms are serialized into ``observations`` in the policy JSON at build time.
    """

    actions: Mapping[str, ActionTermCfg] | None = None
    """Action term configurations (mjlab-compatible).

    Keys are term names (e.g. ``"joint_pos"``).  Values are
    ``ActionTermCfg`` subclass instances serialized into ``actions``
    in the policy JSON at build time.
    """

    terminations: dict[str, TerminationTermCfg] | None = None
    """Termination term configurations (mjlab-compatible).

    Keys are term names (e.g. ``"time_out"``, ``"fallen"``).  Values are
    ``TerminationTermCfg`` instances serialized into ``terminations``
    in the policy JSON at build time.
    """

    policy_joint_names: list[str] | None = None
    """Ordered list of joint names controlled by the policy.

    Required by the browser-side ``PolicyRunner`` to map policy outputs to
    the correct actuators in the MuJoCo model.  When set, serialized as
    ``policy_joint_names`` at the top level of the policy JSON config.
    """

    policy_num_actions: int | None = None
    """Explicit number of policy output actions.

    Use for policies (e.g. muscle-driven) where ``policy_joint_names`` is
    empty and the output size cannot be inferred from joint names.  When set,
    serialized as ``policy_num_actions`` in the policy JSON and used by the
    TS ``PolicyRunner`` instead of ``policy_joint_names.length``.
    """

    default_joint_pos: list[float] | None = None
    """Default joint positions corresponding to ``policy_joint_names``.

    Used by the browser runtime when ``use_default_offset=True``: action=0
    commands this pose.  Must be in the same order as ``policy_joint_names``.
    """

    encoder_bias: list[float] | None = None
    """Per-joint encoder bias corresponding to ``policy_joint_names``.

    Used by the browser runtime to mirror mjlab's joint-position action path:
    the final target written to actuators is ``processed_action - encoder_bias``.
    """

    clip_actions: float | None = None
    """Symmetric bound the raw policy output is clamped to, or ``None`` for unbounded.

    rsl-rl's ``RslRlVecEnvWrapper.step`` clamps to ``[-clip_actions, +clip_actions]``
    *before* ``env.step``, so the action manager — and therefore any ``last_action``
    observation — sees the clamped vector. The browser mirrors that placement: the clamp
    lands on the ONNX output before the action terms or the ``prev_action`` slot read it.

    Not ``ActionTermCfg.clip``, which bounds ``raw * scale + offset`` per target and
    lives on the action term.
    """

    policy_input_shapes: dict[str, list[int]] | None = None
    """Per-input tensor shapes, keyed by ONNX input name, for a graph whose inputs are
    not flat ``[1, N]`` vectors.

    An observation group always produces a flat buffer; a policy whose inputs are
    structured (a look-ahead window, a per-body field) declares the shapes here so the
    runtime can reinterpret that buffer instead of feeding a shape the graph rejects.
    Read off the ONNX itself with :func:`onnx_input_shapes`.
    """

    initial_action: list[float] | None = None
    """Value the stored-action buffer holds before the first inference, or ``None`` for zeros.

    The buffer is what the action terms and any ``prev_action`` observation read, so a
    policy whose output is an *absolute* joint target — not a residual — starts life with
    a whole-pose error unless this is its default pose. Same length as the action vector.
    """

    external_wrench: dict[str, Any] | None = None
    """Bodies an operator can push with a UI command's sliders, for perturbation testing.

    ``{"command_name": str, "targets": [{"body": str, "axes": [x, y, z], "enable": str | None,
    "torque_axes": [x, y, z] | None}]}``, where every string but ``body`` names a UI input on
    that command. Forces are world-frame newtons applied at the body origin -- what
    ``mjData.xfrc_applied`` takes. Distinct from the viewer's mouse-drag forces.
    """

    hand_spring: dict[str, Any] | None = None
    """The virtual Kelvin-Voigt contact a force-exertion policy pushes against.

    Such a policy is trained against a contact anchored at the reference hand: the hand leads, the
    contact resists, and the reaction loads the body. Without it the displaced hand target is simply
    met, nothing is exerted, and a force gauge reads zero however hard the dial is turned. See
    ``core/engine/handSpringContact.ts`` for the shape.
    """

    initial_qpos: list[float] | None = None
    """Optional initial qpos samples or defaults for runtime reset logic."""

    initial_qvel: list[float] | None = None
    """Optional initial qvel samples or defaults for runtime reset logic."""

    extras: dict[str, Any] | None = None
    """Optional extra policy config payload serialized verbatim into JSON."""

    motions: list[MotionConfig] = field(default_factory=list)
    """Reference motions available for this policy."""

    default: bool = False
    """Whether this policy should be the initially selected one in the viewer.

    When multiple policies in a scene have ``default=True``, the first one wins.
    """


class PolicyHandle:
    """Handle for configuring a policy and its commands.

    Commands should be passed via the ``commands=`` parameter of
    :meth:`~mjswan.scene.SceneHandle.add_policy`.

    Example:
        policy = scene.add_policy(
            policy=onnx.load("locomotion.onnx"),
            name="Locomotion",
            config_path="locomotion.json",
            commands={"velocity": mjswan.velocity_command()},
        )
    """

    def __init__(self, policy_config: PolicyConfig, scene: SceneHandle) -> None:
        self._config = policy_config
        self._scene = scene

    @property
    def name(self) -> str:
        """Name of the policy."""
        return self._config.name

    @property
    def model(self) -> onnx.ModelProto:
        """ONNX model for the policy."""
        return self._config.model

    def set_metadata(self, key: str, value: Any) -> PolicyHandle:
        """Set metadata for this policy.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self

    def _append_motion(self, motion: MotionConfig) -> MotionHandle:
        if motion.default:
            for existing in self._config.motions:
                existing.default = False
        self._config.motions.append(motion)
        return MotionHandle(motion, self)

    def add_motion(
        self,
        *,
        name: str,
        source: str,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
        time_source: str = "wall",
    ) -> MotionHandle:
        """Add a bundled ``.npz`` reference motion to this policy.

        ``time_source="sim"`` advances the clip cursor off ``mjData.time`` instead of the
        render loop's ``dt``, so pausing and slow motion keep the reference in step with the
        physics -- which a policy whose look-ahead window is counted in control steps needs.
        """
        motion = MotionConfig(
            name=name,
            source=source,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
            time_source=time_source,  # type: ignore[arg-type]
        )
        return self._append_motion(motion)

    def add_motion_wandb(
        self,
        *,
        name: str | None = None,
        run_path: str | None = None,
        run_id: str | None = None,
        entity: str | None = None,
        project: str | None = None,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
    ) -> MotionHandle:
        """Download a motion artifact from W&B and attach it to this policy."""
        from .wandb_io import fetch_motion_npz_from_wandb_run, resolve_wandb_run_path

        resolved_run_path = resolve_wandb_run_path(
            wandb_run_path=run_path,
            run_id=run_id,
            entity=entity,
            project=project,
        )
        motion_name, payload = fetch_motion_npz_from_wandb_run(resolved_run_path)
        motion = MotionConfig(
            name=name or motion_name,
            data=payload,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
        )
        if default:
            for existing in self._config.motions:
                existing.default = False
        self._config.motions.append(motion)
        return MotionHandle(motion, self)


__all__ = ["PolicyConfig", "PolicyHandle"]
