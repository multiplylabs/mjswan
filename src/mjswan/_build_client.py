"""Automatic Node.js environment setup and client build management.

This module handles:
- Creating isolated Node.js environments using nodeenv
- Installing dependencies
- Building TypeScript/JavaScript clients
- Cross-platform compatibility (Windows/macOS/Linux)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["ClientBuilder", "ensure_node_env", "build_client"]


class ClientBuilder:
    """Manages isolated Node.js environment and client builds."""

    NODE_VERSION = "24.19.0"

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.nodeenv_dir = self.project_dir / ".nodeenv"

    def _get_node_bin(self) -> Path:
        if sys.platform == "win32":
            return self.nodeenv_dir / "Scripts" / "node.exe"
        else:
            return self.nodeenv_dir / "bin" / "node"

    def _get_npm_bin(self) -> Path:
        if sys.platform == "win32":
            return self.nodeenv_dir / "Scripts" / "npm.cmd"
        else:
            return self.nodeenv_dir / "bin" / "npm"

    def _ensure_nodeenv_installed(self) -> None:
        try:
            import nodeenv  # noqa: F401
        except ImportError:
            print("Installing nodeenv...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "nodeenv>=1.9.0"],
                stdout=subprocess.PIPE if not os.getenv("VERBOSE_BUILD") else None,
            )

    def create_env(self, clean: bool = False) -> None:
        if clean and self.nodeenv_dir.exists():
            print(f"Removing existing nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)

        if self.nodeenv_dir.exists():
            node_bin = self._get_node_bin()
            if node_bin.exists():
                try:
                    result = subprocess.run(
                        [str(node_bin), "--version"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        installed_version = result.stdout.strip().lstrip("v")
                        if installed_version == self.NODE_VERSION:
                            print(f"✓ Node.js {self.NODE_VERSION} already available")
                            return
                except Exception as e:
                    print(f"Warning: Could not verify Node.js version: {e}")

        print(f"Creating Node.js {self.NODE_VERSION} environment in {self.nodeenv_dir}")
        self._ensure_nodeenv_installed()

        # Use nodeenv CLI for robustness across versions
        try:
            cmd = [
                sys.executable,
                "-m",
                "nodeenv",
                str(self.nodeenv_dir),
                "--node",
                self.NODE_VERSION,
            ]
            if os.getenv("VERBOSE_BUILD"):
                cmd.append("--verbose")
            subprocess.check_call(cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to create Node.js environment: {e}")

    def install_dependencies(self, clean: bool = False) -> None:
        npm_bin = self._get_npm_bin()
        package_lock = self.project_dir / "package-lock.json"
        node_modules = self.project_dir / "node_modules"

        if clean:
            # Force a fresh install by removing the lock file and node_modules.
            # Useful when switching platforms or resolving corrupted installs.
            if package_lock.exists():
                package_lock.unlink()
            if node_modules.exists():
                shutil.rmtree(node_modules)

        print("Installing npm dependencies (npm install)...")
        subprocess.check_call([str(npm_bin), "install"], cwd=self.project_dir)

    def sync_version_from_python(self) -> None:
        """Sync package.json version with Python package __version__."""
        from mjswan import __version__

        package_json = self.project_dir / "package.json"
        with open(package_json, "r") as f:
            package_data = json.load(f)

        current_version = package_data.get("version", "0.0.0")
        if current_version != __version__:
            print(f"Updating package.json version: {current_version} → {__version__}")
            package_data["version"] = __version__
            # Remove private field if it exists
            package_data.pop("private", None)
            with open(package_json, "w") as f:
                json.dump(package_data, f, indent=2)
                f.write("\n")

    def _has_script(self, script_name: str) -> bool:
        package_json = self.project_dir / "package.json"
        try:
            with open(package_json) as f:
                package_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return script_name in package_data.get("scripts", {})

    def run_build_script(
        self, script_name: str = "build", env: dict[str, str] | None = None
    ) -> None:
        npm_bin = self._get_npm_bin()
        package_json = self.project_dir / "package.json"
        with open(package_json) as f:
            package_data = json.load(f)
        if script_name not in package_data.get("scripts", {}):
            raise ValueError(
                f"Script '{script_name}' not found in {package_json}. "
                f"Available scripts: {list(package_data.get('scripts', {}).keys())}"
            )
        print(f"Running npm script: {script_name}")
        build_env = os.environ.copy()
        if env:
            build_env.update(env)
        subprocess.check_call(
            [str(npm_bin), "run", script_name],
            cwd=self.project_dir,
            env=build_env,
        )

    # Empty stubs for the engine's Custom* registries: author terms compile to a runtime
    # plugins.js instead of inlining, so the SPA stays project-independent.
    _EMPTY_CUSTOM_STUBS = {
        "observation/custom_observations.ts": (
            "// Auto-generated. Custom observations load at runtime via plugins.js"
            " (ADR 0004 §10).\n"
            "export const CustomObservations:"
            " Record<string, new (...args: never[]) => unknown> = {};\n"
        ),
        "command/custom_commands.ts": (
            "// Auto-generated. Custom commands load at runtime via plugins.js"
            " (ADR 0004 §10).\n"
            "import type { CommandTermConstructor } from './types';\n"
            "export const CustomCommands:"
            " Record<string, CommandTermConstructor> = {};\n"
        ),
        "termination/custom_terminations.ts": (
            "// Auto-generated. Custom terminations load at runtime via plugins.js"
            " (ADR 0004 §10).\n"
            "type TerminationConstructor = new ("
            "runner: import('../policy/PolicyRunner').PolicyRunner,"
            " config: import('./TerminationBase').TerminationConfig) =>"
            " import('./TerminationBase').TerminationBase;\n"
            "export const CustomTerminations:"
            " Record<string, TerminationConstructor> = {};\n"
        ),
        "event/custom_events.ts": (
            "// Auto-generated. Custom events load at runtime via plugins.js"
            " (ADR 0004 §10).\n"
            "import type { EventConstructor } from './EventBase';\n"
            "export const CustomEvents: Record<string, EventConstructor> = {};\n"
        ),
    }

    def generate_empty_custom_stubs(self) -> None:
        """Write empty Custom* registry files so the engine compiles project-independently."""
        core = self.project_dir / "src" / "core"
        for rel, content in self._EMPTY_CUSTOM_STUBS.items():
            (core / rel).write_text(content)

    def _plugin_alias_args(self) -> list[str]:
        """esbuild `--alias:` args for bundling author custom-MDP terms.

        - Each `mjswan/<sub>` export → its engine source, so author term files
          (which import base classes from `mjswan/event`, `mjswan/observation`, …)
          resolve deterministically when bundled standalone.
        - `three` → a shim that reuses the engine bundle's single three instance
          at runtime, not a bundled duplicate (so instanceof / shared scene /
          raycasting work across the separately-loaded plugin ESM). See
          src/engine/plugin-three-shim.cjs and src/engine/index.ts.
        """
        package_json = self.project_dir / "package.json"
        with open(package_json) as f:
            exports = json.load(f).get("exports", {})
        args: list[str] = []
        for subpath, target in exports.items():
            if subpath in (".", "./manifest") or not isinstance(target, str):
                continue
            name = "mjswan/" + subpath[len("./") :]
            abs_target = (self.project_dir / target).resolve()
            args.append(f"--alias:{name}={abs_target}")
        three_shim = self.project_dir / "src" / "engine" / "plugin-three-shim.cjs"
        args.append(f"--alias:three={three_shim.resolve()}")
        return args

    @staticmethod
    def _collect_custom_terms() -> dict[str, dict[str, Path]]:
        """Map each MDP kind to {ts_name: source_path} for registered ts_src terms."""
        from mjswan.command import _custom_registry as cmd_reg
        from mjswan.envs.mdp.events import _custom_registry as evt_reg
        from mjswan.envs.mdp.observations import _custom_registry as obs_reg
        from mjswan.envs.mdp.terminations import _custom_registry as term_reg

        kinds = {
            "observations": obs_reg,
            "terminations": term_reg,
            "events": evt_reg,
            "commands": cmd_reg,
        }
        result: dict[str, dict[str, Path]] = {}
        for kind, registry in kinds.items():
            entries: dict[str, Path] = {}
            for sentinel in registry.values():
                ts_src = getattr(sentinel, "ts_src", None)
                ts_name = getattr(sentinel, "ts_name", None)
                if ts_src and ts_name:
                    src = Path(ts_src).expanduser().resolve()
                    if not src.exists():
                        raise FileNotFoundError(
                            f"Custom {kind} ts_src not found: {src}"
                        )
                    entries[ts_name] = src
            if entries:
                result[kind] = entries
        return result

    def build_plugins_module(self, dest: Path) -> bool:
        """Bundle author-supplied custom-MDP terms into a standalone ESM at ``dest``.

        Uses esbuild to inline the terms plus their engine base classes into one
        self-contained module (no bare imports), exporting term constructors
        grouped by kind (``events``/``observations``/``terminations``/``commands``)
        — the ``EnginePlugins`` shape the app hands to ``createEngine`` at load.
        The engine bundle is never rebuilt. Returns False when there are no
        custom terms. Needs Node (esbuild), unlike declarative builds.
        """
        terms = self._collect_custom_terms()
        if not terms:
            return False

        # Generate an entry that imports each term and re-exports it grouped by kind.
        by_src: dict[Path, list[str]] = {}
        for names in terms.values():
            for ts_name, src in names.items():
                by_src.setdefault(src, []).append(ts_name)
        lines = ["// Auto-generated plugin entry — do not edit."]
        for src, names in by_src.items():
            lines.append(
                f"import {{ {', '.join(sorted(set(names)))} }} from {json.dumps(str(src))};"
            )
        for kind, names in terms.items():
            pairs = ", ".join(sorted(names.keys()))
            lines.append(f"export const {kind} = {{ {pairs} }};")
        entry = self.project_dir / "src" / ".plugins-entry.ts"
        entry.write_text("\n".join(lines) + "\n")

        esbuild = self.project_dir / "node_modules" / ".bin" / "esbuild"
        if not esbuild.exists():
            # Custom-JS builds need Node; install if a cached SPA skipped it.
            self.create_env()
            self.install_dependencies()

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(
                [
                    str(esbuild),
                    str(entry),
                    "--bundle",
                    "--format=esm",
                    "--platform=browser",
                    f"--outfile={dest}",
                    *self._plugin_alias_args(),
                ],
                cwd=self.project_dir,
            )
        finally:
            entry.unlink(missing_ok=True)
        return True

    def generate_viewer_config_defaults(self) -> None:
        """Generate viewer_config_defaults.ts from Python ViewerConfig defaults."""
        from mjswan.viewer import ViewerConfig

        d = ViewerConfig()
        fovy_default = 45  # Python fovy=None means "use 45 degrees"

        lines = [
            "// Auto-generated from Python ViewerConfig defaults. Do not edit manually.",
            "// Regenerated by mjswan._build_client.ClientBuilder.generate_viewer_config_defaults()",
            "",
            "export const VIEWER_CONFIG_DEFAULTS = {",
            f"  lookat: [{d.lookat[0]}, {d.lookat[1]}, {d.lookat[2]}] as [number, number, number],",
            f"  distance: {d.distance},",
            f"  elevation: {d.elevation},",
            f"  azimuth: {d.azimuth},",
            f"  fovy: {fovy_default},",
            f"  originType: '{d.origin_type.name}' as const,",
            f"  enableReflections: {str(d.enable_reflections).lower()},",
            f"  enableShadows: {str(d.enable_shadows).lower()},",
            f"  handTracking: {str(d.hand_tracking).lower()},",
            f"  height: {d.height},",
            f"  width: {d.width},",
            "} as const;",
            "",
        ]
        output_path = (
            self.project_dir / "src" / "core" / "engine" / "viewer_config_defaults.ts"
        )
        output_path.write_text("\n".join(lines))

    def _build_meta(
        self, base_path: str, gtm_id: str | None, mt: bool, debug: bool
    ) -> dict[str, object]:
        """Cache key for a built SPA: version + build opts + a source fingerprint.

        The SPA is project-independent (custom terms load at runtime via
        plugins.js, ADR 0004 §10), so any build with a matching key is reusable.
        The fingerprint rebuilds the SPA whenever the app codebase changes.
        """
        from mjswan import __version__

        return {
            "version": __version__,
            "base_path": base_path,
            "gtm_id": gtm_id,
            "mt": mt,
            "debug": debug,
            "source": self._source_fingerprint(),
        }

    # Derived files, excluded from the fingerprint: regenerated identically each build.
    _GENERATED_TS = frozenset(
        {
            "src/core/observation/custom_observations.ts",
            "src/core/command/custom_commands.ts",
            "src/core/event/custom_events.ts",
            "src/core/termination/custom_terminations.ts",
            "src/core/engine/viewer_config_defaults.ts",
        }
    )

    def _source_fingerprint(self) -> str:
        """SHA-256 over the frontend build inputs (app source + config + declared deps).

        Invalidates the cache whenever the app codebase changes. Excludes derived
        outputs (node_modules / dist / .nodeenv), the auto-generated stubs above,
        and package-lock.json (npm may rewrite it); package.json's ``version`` is
        normalized out since it is already a separate part of the key.
        """
        inputs: list[Path] = []
        for top in (
            "package.json",
            "vite.config.ts",
            "vite.lib.config.ts",
            "tsconfig.json",
            "index.html",
        ):
            p = self.project_dir / top
            if p.exists():
                inputs.append(p)
        for root in ("src", "public"):
            base = self.project_dir / root
            if base.exists():
                inputs.extend(p for p in base.rglob("*") if p.is_file())

        digest = hashlib.sha256()
        for path in sorted(
            inputs, key=lambda p: p.relative_to(self.project_dir).as_posix()
        ):
            rel = path.relative_to(self.project_dir).as_posix()
            if rel in self._GENERATED_TS:
                continue
            digest.update(rel.encode())
            if rel == "package.json":
                data = json.loads(path.read_text())
                data.pop("version", None)
                digest.update(json.dumps(data, sort_keys=True).encode())
            else:
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _cached_spa_matches(self, meta: dict[str, object]) -> bool:
        dist = self.project_dir / "dist"
        marker = dist / ".mjswan-build-meta.json"
        if not (dist / "index.html").exists() or not marker.exists():
            return False
        try:
            return json.loads(marker.read_text()) == meta
        except (OSError, json.JSONDecodeError):
            return False

    def build(
        self,
        *,
        clean: bool = False,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
        build_frontend: bool | None = None,
    ) -> None:
        """Build the standalone SPA into ``dist/``.

        ``build_frontend``: True forces a build; False requires a matching cached
        artifact (raises otherwise); None (default) reuses the cache when it
        matches and builds only when it doesn't. The SPA is project-independent,
        so the cache is keyed on the mjswan version + base_path/gtm_id/mt/debug.
        """
        meta = self._build_meta(base_path, gtm_id, mt, debug)
        if not clean and build_frontend is not True and self._cached_spa_matches(meta):
            print("✓ Reusing cached frontend build (dist/)")
            return
        if build_frontend is False:
            raise RuntimeError(
                "build_frontend=False but no matching prebuilt dist/ was found."
            )
        try:
            self.create_env(clean=clean)
            self.sync_version_from_python()
            # Custom terms are runtime plugins now, so the engine stubs stay empty.
            self.generate_empty_custom_stubs()
            self.generate_viewer_config_defaults()
            self.install_dependencies(clean=clean)
            env: dict[str, str] = {"MJSWAN_BASE_PATH": base_path}
            if gtm_id:
                env["MJSWAN_GTM_ID"] = gtm_id
            if mt:
                env["MJSWAN_MT"] = "1"
            if debug:
                env["MJSWAN_DEBUG"] = "1"
            # The standalone app needs only the SPA build; the library build (`mjswan.js`, for
            # mjswan Cloud) comes from the full `build` script during npm publish.
            script = "build:spa" if self._has_script("build:spa") else "build"
            self.run_build_script(script, env=env)
            (self.project_dir / "dist" / ".mjswan-build-meta.json").write_text(
                json.dumps(meta)
            )
            print("✓ Build completed successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build failed with exit code {e.returncode}") from e
        except Exception as e:
            raise RuntimeError(f"Build failed: {e}") from e

    def cleanup(self) -> None:
        if self.nodeenv_dir.exists():
            print(f"Cleaning up nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)


def ensure_node_env(project_dir: Path, clean: bool = False) -> Path:
    builder = ClientBuilder(project_dir)
    builder.create_env(clean=clean)
    return builder.nodeenv_dir


def build_client(
    project_dir: Path,
    clean: bool = False,
    script: str = "build",
    base_path: str = "/",
    mt: bool = False,
) -> None:
    builder = ClientBuilder(project_dir)
    builder.build(clean=clean, base_path=base_path, mt=mt)
