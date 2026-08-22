"""Builder class for constructing mjswan applications.

This module provides the main Builder class which serves as the entry point
for programmatically creating interactive MuJoCo simulations.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import mujoco
import onnx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from . import __version__
from ._build_client import ClientBuilder
from .app import MjswanApp
from .envs.mdp.actions.actions import (
    MuscleActivationActionCfg,
    validate_muscle_actuators,
)
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig
from .splat import SplatConfig
from .utils import collect_spec_assets, name2id, to_zip_deflated


def _build_uses_custom_js() -> bool:
    """Whether the current build embeds author-supplied TypeScript.

    Surfaced at the top of ``config.json`` so a consumer can enforce a
    declarative-only policy without inspecting the bundled engine (ADR 0003).
    """
    from .command import _custom_registry as _command_registry
    from .envs.mdp.events import _custom_registry as _event_registry
    from .envs.mdp.observations import _custom_registry as _obs_registry
    from .envs.mdp.terminations import _custom_registry as _term_registry

    for registry in (_obs_registry, _term_registry, _event_registry, _command_registry):
        for sentinel in registry.values():
            if getattr(sentinel, "ts_src", None) is not None:
                return True
    return False


def _scene_trace_env(scene: SceneConfig) -> Any | None:
    """The env ONNX tracing runs term bodies against, built on first use.

    Deferred to build time so a tracking task's env is constructed only once its clip is
    on disk, and so a scene that traces nothing never pays for one at all.
    """
    if scene.mjlab_env is not None:
        return scene.mjlab_env
    if scene.mjlab_env_cfg is None:
        return None
    from .trace_env import build_mjlab_env

    scene.mjlab_env = build_mjlab_env(scene.mjlab_env_cfg)
    scene.mjlab_env.reset()
    return scene.mjlab_env


def _motion_key(motion: Any) -> str:
    """Identity of a clip's content, so two policies sharing one share its file."""
    if motion.data is not None:
        return hashlib.sha256(motion.data).hexdigest()
    if motion.source is not None:
        src = _resolve_motion_source(motion.source)
        try:
            return hashlib.sha256(src.read_bytes()).hexdigest()
        except OSError:
            return f"src:{src}"  # missing; the copy below warns about it
    return f"empty:{motion.name}"


def _resolve_motion_source(source: str) -> Path:
    src = Path(source).expanduser()
    return src if src.is_absolute() else (Path.cwd() / src).resolve()


def _write_scene_motions(scene: SceneConfig, scene_dir: Path) -> dict[str, str]:
    """Write each distinct clip in the scene once; return content key -> filename.

    Scene-scoped rather than per-policy: the checkpoints of one run share a clip, and a
    copy per policy meant N copies of the same megabytes. Same name and same content
    collapse to one file; same name and different content get a ``_1``/``_2`` suffix.
    """
    files: dict[str, str] = {}
    used: set[str] = set()
    for policy in scene.policies:
        for motion in policy.motions:
            key = _motion_key(motion)
            if key in files:
                continue
            stem = name2id(motion.name)
            filename = f"{stem}.npz"
            suffix = 0
            while filename in used:
                suffix += 1
                filename = f"{stem}_{suffix}.npz"
            used.add(filename)
            files[key] = filename

            target = scene_dir / filename
            if motion.data is not None:
                target.write_bytes(motion.data)
            elif motion.source is not None:
                src = _resolve_motion_source(motion.source)
                if src.exists():
                    shutil.copy2(str(src), str(target))
                else:
                    warnings.warn(
                        f"Motion source file not found: {src}",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
    return files


def _point_env_cfg_at_bundled_motion(
    scene: SceneConfig, scene_dir: Path, motion_files: dict[str, str]
) -> None:
    """Aim a tracking task's ``motion_file`` at the clip just written to the bundle.

    mjlab registers tracking tasks with ``motion_file=""`` and ``MotionLoader`` reads the
    path when the env is constructed, so the bundled copy is what the trace env loads —
    no second copy anywhere.
    """
    env_cfg = scene.mjlab_env_cfg
    if env_cfg is None or scene.mjlab_env is not None or not motion_files:
        return
    for term in (getattr(env_cfg, "commands", None) or {}).values():
        if not hasattr(term, "motion_file"):
            continue
        existing = getattr(term, "motion_file", "") or ""
        if existing and Path(existing).expanduser().is_file():
            continue
        term.motion_file = str(scene_dir / next(iter(motion_files.values())))
        return


def _require_control_dt(scene: Any) -> float:
    """A scene's seconds-per-control-step, or fail naming the scene.

    Raises rather than defaulting, because a wrong rate raises no error at playback: the
    substep count, resample schedule, and interval timers all just count in the wrong
    unit, and it reads as a policy that behaves badly.

    Only reached for a scene carrying a policy; a viewer-only scene has no rate.
    """
    if scene.control_dt is None:
        raise ValueError(
            f"Scene {scene.name!r} has policies but no control_dt. Pass it to "
            "add_scene(control_dt=...) as seconds per control step — mjlab's "
            "`timestep * decimation`, the rate the policy was trained to act at. The "
            "model carries only the physics timestep, so this cannot be inferred, and "
            "guessing it wrong is silent. add_scene_mjlab() fills it in from the task."
        )
    if scene.control_dt <= 0:
        raise ValueError(
            f"Scene {scene.name!r} has control_dt={scene.control_dt!r}; it must be a "
            "positive number of seconds."
        )
    return float(scene.control_dt)


class _SceneSteps:
    """The sub-step line under a project's progress bar: one phase at a time, counting
    its own units and naming the one it is on."""

    def __init__(self, progress: Progress):
        self._progress = progress
        self._task = progress.add_task("", total=1, scene="")
        self._started = False

    def begin(self, label: str, total: int) -> None:
        self._started = False
        # `total=None` means "keep the current total" to rich, so it is always passed.
        self._progress.reset(
            self._task, total=total, description=f"   └ {label}", scene=""
        )

    def on(self, name: str) -> None:
        """Name the unit now starting; the previous one counts as done."""
        self._progress.update(self._task, scene=name, advance=1 if self._started else 0)
        self._started = True

    def close(self) -> None:
        self._progress.remove_task(self._task)


class Builder:
    """Builder for creating mjswan applications.

    The Builder class provides a fluent API for programmatically constructing
    interactive MuJoCo simulations with ONNX policies. It handles projects, scenes, and policies hierarchically.
    """

    def __init__(
        self,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize a new Builder instance.

        Args:
            base_path: Base path for subdirectory deployment (e.g., '/mjswan/').
            gtm_id: Google Tag Manager ID (e.g., 'GTM-XXXXXXX'). Injects GTM snippet if set.
            mt: Enable multi-threaded MuJoCo WASM. Requires COOP/COEP headers — these are
                written as a ``_headers`` file (Netlify/Cloudflare Pages/Vercel) and a
                service worker (required for GitHub Pages hosting). Defaults to False.
            debug: Keep browser console messages in the built app. Defaults to False
                (console messages are stripped from the production bundle).
        """
        self._projects: list[ProjectConfig] = []
        self._base_path = base_path
        self._gtm_id = gtm_id
        self._mt = mt
        self._debug = debug

    @classmethod
    def from_mjlab(
        cls,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        project_name: str = "mjlab",
        play: bool | None = None,
        env_cfg: Any | None = None,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> Builder:
        """Create a Builder pre-configured with a single mjlab task.

        This is a convenience factory for the common pattern of visualizing one
        mjlab task. The returned Builder can be further modified before calling
        :meth:`build`.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all ``model_*.pt``
                checkpoints from each run are fetched and converted to ONNX
                via mjlab+torch (both required). ``task_id`` above is reused
                for the conversion. Defaults to ``None`` (no policy attached).
                For finer control (e.g. ``only_latest=True``, custom
                observations/actions), build manually with
                :meth:`add_project` → :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`
                → :meth:`~mjswan.scene.SceneHandle.add_policy_wandb`.
            project_name: Name for the auto-created project. Defaults to ``"mjlab"``.
            play: Which of the task's two registered configs to load; unset means play.
                Mutually exclusive with ``env_cfg``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            env_cfg: Pre-loaded (and possibly edited) env config, for a task whose
                registered one is incomplete — mjlab's tracking tasks ship
                ``commands["motion"].motion_file = ""``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            base_path: Base path for the application (e.g., ``"/mjswan/"``).
            gtm_id: Optional Google Tag Manager container ID.

        Returns:
            Builder with one project and one scene already configured.

        Example:
            ```python
            # Minimal usage
            app = mjswan.Builder.from_mjlab("go2_flat").build()
            app.launch()

            # Load mjlab scene and attach all checkpoints from a W&B run
            app = mjswan.Builder.from_mjlab(
                "Mjlab-Velocity-Flat-Anymal-C",
                run_path="ttktjmt-org/mjlab/dqxvf0eb",
            ).build()

            # Customise before building
            builder = mjswan.Builder.from_mjlab("go2_flat")
            scene = builder.get_projects()[0].scenes[0]  # access SceneConfig
            app = builder.build()
            ```
        """
        builder = cls(base_path=base_path, gtm_id=gtm_id, mt=mt, debug=debug)
        builder.add_project_mjlab(
            task_id,
            run_path=run_path,
            project_name=project_name,
            play=play,
            env_cfg=env_cfg,
        )
        return builder

    def add_project_mjlab(
        self,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        project_name: str = "mjlab",
        play: bool | None = None,
        env_cfg: Any | None = None,
    ) -> ProjectHandle:
        """Add a project pre-configured with a single mjlab task.

        Convenience for the common pattern of visualizing one mjlab task:
        creates a project, adds the mjlab scene, and (optionally) attaches all
        ``model_*.pt`` checkpoints from one or more W&B runs as ONNX policies.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all checkpoints are fetched
                and converted to ONNX via mjlab+torch (both required) using
                ``task_id``. Defaults to ``None`` (no policy attached).
            project_name: Name for the created project. Defaults to ``"mjlab"``.
            play: Which of the task's two registered configs to load; unset means play.
                Mutually exclusive with ``env_cfg``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            env_cfg: Pre-loaded (and possibly edited) env config. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.

        Returns:
            ProjectHandle for the created project.
        """
        project = self.add_project(name=project_name)
        # `play` stays unresolved: `add_scene_mjlab` rejects it alongside `env_cfg`, so
        # materialising the default here would trip that guard for every caller.
        scene = project.add_scene_mjlab(task_id, play=play, env_cfg=env_cfg)
        if run_path is not None:
            scene.add_policy_wandb(run_path, task_id=task_id)
        return project

    def add_project(self, name: str, *, id: str | None = None) -> ProjectHandle:
        """Add a new project to the builder.

        Args:
            name: Name for the project (displayed in the UI).
            id: Optional ID for URL routing. If not provided, the first project
                defaults to None (main route), and subsequent projects default to sanitized name.

        Returns:
            ProjectHandle for adding scenes and further configuration.
        """
        # Project ID: explicit > None for first project (main route) > sanitized name
        if id is not None:
            project_id = id
        elif not self._projects:
            project_id = None
        else:
            project_id = name2id(name)

        project = ProjectConfig(name=name, id=project_id)
        self._projects.append(project)
        return ProjectHandle(project, self)

    def build(
        self,
        output_dir: str | Path | None = None,
        build_frontend: bool | None = None,
    ) -> MjswanApp:
        """Build the application from the configured projects.

        This method finalizes the configuration and creates a MjswanApp
        instance. If output_dir is provided, it also saves the application
        to that directory. If output_dir is not provided, it defaults to
        'dist' in the caller's directory.

        Args:
            output_dir: Optional directory to save the application files.
                       If None, defaults to 'dist' in the caller's directory.

        Returns:
            MjswanApp instance ready to be launched.
        """
        if not self._projects:
            raise ValueError(
                "Cannot build an empty application. "
                "You must add at least one project using builder.add_project() before building.\n"
                "Example:\n"
                "  builder = mwx.Builder()\n"
                "  project = builder.add_project(name='My Project')\n"
                "  scene = project.add_scene(spec=mujoco_spec, name='Scene 1')\n"
                "  app = builder.build()"
            )

        # Get caller's file path
        frame = inspect.stack()[1]
        caller_file = frame.filename
        # Handle REPL or interactive mode where filename might be <stdin> or similar
        if caller_file.startswith("<") and caller_file.endswith(">"):
            base_dir = Path.cwd()
        else:
            base_dir = Path(caller_file).parent

        if output_dir is None:
            output_path = base_dir / "dist"
        else:
            # Resolve relative paths against the caller's directory
            output_path = base_dir / Path(output_dir)

        # TODO: Build with separate function (and then save the web app with _save_web). And set scene.path and policy.path after building.
        self._save_web(output_path, build_frontend=build_frontend)

        return MjswanApp(output_path)

    def _save_config_json(self, output_path: Path) -> None:
        """Save configuration as JSON.

        Creates root assets/config.json with project metadata and structure information.
        Individual project assets (scenes/policies) are saved under project-id/assets/.
        """
        # Create root config with project metadata and structure info
        uses_custom_js = _build_uses_custom_js()
        root_config = {
            "version": __version__,
            "uses_custom_js": uses_custom_js,
            # Author custom-MDP terms, loaded by the app in trusted contexts only.
            **({"plugins": "assets/plugins.js"} if uses_custom_js else {}),
            "projects": [
                {
                    "name": project.name,
                    "id": project.id,
                    "scenes": [
                        {
                            "name": scene.name,
                            "path": f"{name2id(scene.name)}/{scene.scene_filename}",
                            **(
                                {
                                    "splats": [
                                        self._build_splat_config_dict(scene, s)
                                        for s in scene.splats
                                    ]
                                }
                                if scene.splats
                                else {}
                            ),
                            **(
                                {"splatSection": True}
                                if scene.splat_section and not scene.splats
                                else {}
                            ),
                            **(
                                {"camera": scene.viewer.to_dict()}
                                if scene.viewer and scene.viewer.to_dict()
                                else {}
                            ),
                            **({"events": scene.events} if scene.events else {}),
                            **(
                                {"terrainData": scene.terrain_data}
                                if scene.terrain_data
                                else {}
                            ),
                            **(
                                {"controlDt": _require_control_dt(scene)}
                                if scene.policies
                                else {}
                            ),
                            "policies": [
                                (
                                    {
                                        "name": policy.name,
                                        **(
                                            {
                                                "config": f"{name2id(scene.name)}/"
                                                f"{name2id(policy.name)}.json"
                                            }
                                            if getattr(policy, "config_path", None)
                                            or getattr(policy, "commands", None)
                                            or getattr(policy, "observations", None)
                                            or getattr(policy, "actions", None)
                                            or getattr(policy, "terminations", None)
                                            or getattr(policy, "motions", None)
                                            else {}
                                        ),
                                        **(
                                            {"source": policy.source_path}
                                            if getattr(policy, "source_path", None)
                                            else {}
                                        ),
                                        **(
                                            {"default": True}
                                            if getattr(policy, "default", False)
                                            else {}
                                        ),
                                        **(
                                            {
                                                "motions": [
                                                    motion.to_summary_dict()
                                                    for motion in policy.motions
                                                ]
                                            }
                                            if getattr(policy, "motions", None)
                                            else {}
                                        ),
                                    }
                                )
                                for policy in scene.policies
                            ],
                        }
                        for scene in project.scenes
                    ],
                }
                for project in self._projects
            ],
        }

        # Save root config.json in assets directory
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)
        root_config_file = assets_dir / "config.json"
        with open(root_config_file, "w") as f:
            json.dump(root_config, f, indent=2)

    def _save_mt_headers(self, output_path: Path) -> None:
        """Write COOP/COEP response headers needed by multi-threaded MuJoCo.

        Two mechanisms are written so the output works on any static host:
        - ``_headers``: honored by Netlify, Cloudflare Pages, and Vercel.
        - ``coi-serviceworker.js`` (emitted by the Vite build only when mt=True): used by the
          injected inline script for GitHub Pages, which cannot set response headers.
        """
        headers_content = (
            "/*\n"
            "  Cross-Origin-Opener-Policy: same-origin\n"
            "  Cross-Origin-Embedder-Policy: require-corp\n"
            "\n"
        )
        (output_path / "_headers").write_text(headers_content)

    def _build_splat_config_dict(self, scene: SceneConfig, splat: SplatConfig) -> dict:
        """Build the splat dict for config.json.

        When ``source`` is set the file is copied to the scene asset directory
        during :meth:`_save_web`, and the resulting relative path is injected
        here so the frontend can resolve it to a URL.
        """
        d = splat.to_dict()
        if splat.source is not None:
            d["path"] = f"{name2id(scene.name)}/{name2id(splat.name)}.spz"
        return d

    def _policy_filename(self, name: str) -> str:
        if not name or name.strip() == "":
            raise ValueError("Policy name must be a non-empty string.")
        if "/" in name or "\\" in name:
            raise ValueError(
                "Policy name cannot contain path separators ('/' or '\\')."
            )
        return name

    def _serialize_policy_config(
        self,
        policy,
        env,
        scene_dir: Path,
        policy_path: Path,
        motion_files: dict[str, str] | None = None,
    ) -> dict | None:
        """Assemble one policy's JSON config, tracing ONNX terms via the scene's env.

        Returns ``None`` (with a warning) when ``policy.config_path`` names a file that
        does not exist, or when the policy has nothing to serialize. A trace or parse
        failure is not caught: it fails the build.
        """
        from ._onnx_build import (
            policy_native_sizes,
            serialize_command,
            serialize_observation_group,
            serialize_terminations,
        )

        config_path = getattr(policy, "config_path", None)
        has_mdp = (
            policy.commands
            or policy.observations
            or policy.actions
            or policy.terminations
            or policy.policy_joint_names
            or policy.policy_num_actions
            or policy.motions
            or policy.clip_actions is not None
        )
        if not config_path and not has_mdp:
            return None

        if env is None and (policy.observations or policy.terminations):
            raise ValueError(
                f"Policy {policy.name!r} on scene {scene_dir.name!r} has "
                "observation/termination terms to trace, but the scene has no trace "
                "env to trace them against. `add_scene_mjlab` supplies one; a plain "
                "`add_scene` scene needs it explicitly: "
                "`scene.set_trace_env(build_single_entity_trace_env(spec_fn))` "
                "(ADR 0005 §6)."
            )

        data: dict = {}
        if config_path:
            config_src = Path(config_path).expanduser()
            if not config_src.is_absolute():
                config_src = (Path.cwd() / config_src).resolve()
            if not config_src.exists():
                warnings.warn(
                    f"Policy config path not found: {config_src}",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                return None
            with open(config_src, "r") as f:
                data = json.load(f)
            data.setdefault("onnx", {})
            if isinstance(data["onnx"], dict):
                onnx_config = data["onnx"]
                onnx_config["path"] = policy_path.name
                meta = dict(onnx_config.get("meta") or {})
                if "in_keys" in data and "in_keys" not in meta:
                    meta["in_keys"] = data["in_keys"]
                if "out_keys" in data and "out_keys" not in meta:
                    meta["out_keys"] = data["out_keys"]
                if meta:
                    onnx_config["meta"] = meta
        else:
            data = {"onnx": {"path": policy_path.name}}

        if policy.policy_joint_names:
            data["policy_joint_names"] = policy.policy_joint_names
        if policy.policy_num_actions:
            data["policy_num_actions"] = policy.policy_num_actions
        if policy.default_joint_pos:
            data["default_joint_pos"] = policy.default_joint_pos
        if policy.encoder_bias:
            data["encoder_bias"] = policy.encoder_bias
        # Not `if policy.clip_actions:` — 0.0 is a legal bound, not "unset".
        if policy.clip_actions is not None:
            data["clip_actions"] = float(policy.clip_actions)
        if getattr(policy, "policy_input_shapes", None):
            data["policy_input_shapes"] = {
                name: [int(dim) for dim in shape]
                for name, shape in policy.policy_input_shapes.items()
            }
        if getattr(policy, "initial_action", None):
            data["initial_action"] = [float(v) for v in policy.initial_action]
        if getattr(policy, "external_wrench", None):
            data["external_wrench"] = policy.external_wrench
        if getattr(policy, "hand_spring", None):
            data["hand_spring"] = policy.hand_spring
        if getattr(policy, "initial_qpos", None):
            data["initial_qpos"] = policy.initial_qpos
        if getattr(policy, "initial_qvel", None):
            data["initial_qvel"] = policy.initial_qvel
        if getattr(policy, "extras", None):
            data["extras"] = policy.extras

        if policy.commands:
            data["commands"] = {
                name: serialize_command(name, cmd, env, scene_dir)
                for name, cmd in policy.commands.items()
            }
        if policy.observations:
            native_sizes = policy_native_sizes(data, policy.commands)
            obs_config = data.get("observations", {})
            for key, group in policy.observations.items():
                # Never overwrite a group `config_path` already declared. The key names
                # the fused graph too, so two groups cannot collide.
                target_key = key
                if target_key in obs_config:
                    target_key = f"{key}_monitor"
                obs_config[target_key] = serialize_observation_group(
                    group, env, scene_dir, target_key, native_sizes
                )
            data["observations"] = obs_config
        if policy.actions:
            # Merged field-wise over the authored config, where a motor robot's PD gains
            # live, so a scene can tweak the offset without restating them.
            authored = data.get("actions", {})
            data["actions"] = {
                name: {**authored.get(name, {}), **cfg.to_dict()}
                for name, cfg in policy.actions.items()
            }
        if policy.terminations:
            terminations = serialize_terminations(policy.terminations, env, scene_dir)
            if terminations:
                data["terminations"] = terminations
        if policy.motions:
            files = motion_files or {}
            data["motions"] = [
                motion.to_dict(files[_motion_key(motion)]) for motion in policy.motions
            ]
        return data

    def _validate_muscle_action_terms(self, scene: SceneConfig) -> None:
        """Validate every ``MuscleActivationActionCfg`` in the scene's policies.

        Each term's ``actuator_names`` must resolve to muscle-dyntype actuators
        in the scene's MuJoCo model. Raises ``ValueError`` on the first violation
        so users see configuration mistakes before deployment rather than at
        runtime in the browser.
        """
        muscle_terms: list[tuple[str, MuscleActivationActionCfg]] = []
        for policy in scene.policies:
            actions = getattr(policy, "actions", None) or {}
            for term_name, cfg in actions.items():
                if isinstance(cfg, MuscleActivationActionCfg):
                    muscle_terms.append((term_name, cfg))
        if not muscle_terms:
            return

        model = scene.model
        if model is None and scene.spec is not None:
            model = scene.spec.compile()
        if model is None:
            return

        for term_name, cfg in muscle_terms:
            validate_muscle_actuators(model, cfg, term_name=term_name)

    def _save_web(self, output_path: Path, build_frontend: bool | None = None) -> None:
        """Save as a complete web application.

        Output structure:
            dist/
            ├── index.html
            ├── logo.svg
            ├── robots.txt
            ├── assets/
            │   ├── config.json
            │   └── (compiled js/css files)
            └── <project-id>/ (or 'main')
                ├── index.html
                ├── logo.svg
                └── assets/
                    └── <scene-id>/
                        ├── scene.mjz/.mjb
                        ├── <policy-id>.onnx
                        ├── <policy-id>.json
                        └── <splat-id>.spz  (when local source provided)
        """
        if output_path.exists():
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Copy template directory
        template_dir = Path(__file__).parent / "template"
        client_builder: ClientBuilder | None = None
        if template_dir.exists():
            # Build client first (reuses a cached SPA build when it matches)
            package_json = template_dir / "package.json"
            if package_json.exists():
                print("Building the mjswan application...")
                client_builder = ClientBuilder(template_dir)
                client_builder.build(
                    base_path=self._base_path,
                    gtm_id=self._gtm_id,
                    mt=self._mt,
                    debug=self._debug,
                    build_frontend=build_frontend,
                )

            # Only the built SPA, so dev scaffolding stays out by construction.
            built_dist = template_dir / "dist"
            if built_dist.is_dir():
                # Dev-only artifacts vite emits: the E2E fixture, the build-cache key.
                spa_excludes = {"fixtures", ".mjswan-build-meta.json"}
                for item in built_dist.iterdir():
                    if item.name in spa_excludes:
                        continue
                    dest = output_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                # Ship the license alongside the app.
                license_file = template_dir / "LICENSE"
                if license_file.exists():
                    shutil.copy2(license_file, output_path / license_file.name)
            else:
                warnings.warn(
                    f"No built SPA found at {built_dist}; the output will be "
                    "missing the web application.",
                    category=RuntimeWarning,
                )
        else:
            warnings.warn(
                f"Template directory not found at {template_dir}.",
                category=RuntimeWarning,
            )

        # Create root assets directory for shared config
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Author custom-MDP terms compile to a runtime ESM beside config.json, via esbuild.
        if _build_uses_custom_js() and client_builder is not None:
            print("Compiling custom-MDP term module (plugins.js)...")
            client_builder.build_plugins_module(output_path / "assets" / "plugins.js")

        # Write COOP/COEP headers for multi-threaded MuJoCo (SharedArrayBuffer)
        if self._mt:
            self._save_mt_headers(output_path)

        # Save MuJoCo models and ONNX policies per project
        max_name_len = max(len(p.name) for p in self._projects)
        for project in self._projects:
            # Use 'main' for projects without ID, otherwise use the project ID
            project_dir_name = project.id if project.id else "main"
            project_dir = output_path / project_dir_name
            project_assets_dir = project_dir / "assets"

            # Create directories
            project_assets_dir.mkdir(parents=True, exist_ok=True)

            # Copy index.html to each project directory so direct navigation works
            root_index = output_path / "index.html"
            if root_index.exists():
                shutil.copy(str(root_index), str(project_dir / "index.html"))

            # Copy static root assets
            for static_name in ["logo.svg"]:
                src_static = output_path / static_name
                if src_static.exists():
                    shutil.copy(str(src_static), str(project_dir / static_name))

            # Save scenes and policies
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[dim]{task.fields[scene]}"),
            ) as progress:
                task = progress.add_task(
                    project.name.ljust(max_name_len),
                    total=len(project.scenes),
                    scene="",
                )
                steps = _SceneSteps(progress)
                for scene in project.scenes:
                    progress.update(task, scene=scene.name)
                    scene_id = name2id(scene.name)
                    scene_dir = project_assets_dir / scene_id
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    scene_path = scene_dir / scene.scene_filename

                    # First: the conversion is what creates this scene's policies (and
                    # hands over its export env as the trace env).
                    for pending in scene.pending_conversions:
                        steps.begin(
                            "converting checkpoints", total=len(pending.run_paths)
                        )
                        pending.run(steps.on)
                    scene.pending_conversions.clear()

                    self._validate_muscle_action_terms(scene)

                    steps.begin("building scene", total=3)
                    steps.on("packaging scene")
                    if scene.spec is not None:
                        scene.spec.assets.update(collect_spec_assets(scene.spec))
                        to_zip_deflated(scene.spec, str(scene_path))  # Saves as .mjz
                        scene.spec = None
                    else:
                        if scene.model is None:
                            raise RuntimeError(
                                f"Scene '{scene.name}' has no model to save as .mjb"
                            )
                        mujoco.mj_saveModel(
                            scene.model, str(scene_path)
                        )  # Saves as .mjb
                        scene.model = None
                    gc.collect()

                    # Before anything needing the trace env: a tracking task's env loads
                    # its clip from disk, and the bundled copy is that file.
                    steps.on("bundling clips")
                    motion_files = _write_scene_motions(scene, scene_dir)
                    _point_env_cfg_at_bundled_motion(scene, scene_dir, motion_files)

                    steps.on("copying splats")
                    for splat in scene.splats:
                        if splat.source is not None:
                            src = Path(splat.source).expanduser()
                            if not src.is_absolute():
                                src = (Path.cwd() / src).resolve()
                            if src.exists():
                                shutil.copy2(
                                    str(src),
                                    str(scene_dir / f"{name2id(splat.name)}.spz"),
                                )
                            else:
                                warnings.warn(
                                    f"Splat source file not found: {src}",
                                    category=RuntimeWarning,
                                    stacklevel=2,
                                )

                    steps.begin(
                        "tracing mdps",
                        total=len(scene.events or {}) + len(scene.policies),
                    )
                    if scene.events:
                        from ._onnx_build import serialize_events

                        # Overwrites `scene.events` with the JSON list `_save_config_json`
                        # reads after this loop.
                        scene.events = serialize_events(
                            scene.events,
                            _scene_trace_env(scene),
                            scene_dir,
                            on_term=lambda name: steps.on(f"event/{name}"),
                        )

                    for policy in scene.policies:
                        steps.on(f"policy/{policy.name}")
                        policy_id = name2id(policy.name)
                        policy_path = scene_dir / f"{policy_id}.onnx"
                        onnx.save(policy.model, str(policy_path))

                        data = self._serialize_policy_config(
                            policy,
                            _scene_trace_env(scene),
                            scene_dir,
                            policy_path,
                            motion_files,
                        )
                        if data is not None:
                            target = policy_path.with_suffix(".json")
                            with open(target, "w") as f:
                                json.dump(data, f, indent=2)

                    progress.advance(task)
                steps.close()

        # After the scene loop, which resolves each scene's `events` into the JSON form.
        self._save_config_json(output_path)

        print(f"✓ Saved mjswan application to: {output_path}")

    def get_projects(self) -> list[ProjectConfig]:
        """Get a copy of all project configurations.

        Returns:
            List of ProjectConfig objects.
        """
        return self._projects.copy()


__all__ = ["Builder"]
