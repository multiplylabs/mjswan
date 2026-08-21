# VR hand tracking as injected mocap bodies

> Status: **Accepted (implemented)** — extends the viewer layer only. The XR hands
> are a *viewer* disturbance in the same sense as ADR 0004's mouse drag: they enter
> the simulation through `mocap_pos` / `mocap_quat` and an equality constraint, never
> through the MDP. No observation, action, termination, or command surface changes.

## Context

The viewer already entered VR — `renderer.xr.enabled`, a `VRButton`, and a WebXR
session that renders the scene in stereo. What it could not do was let the viewer
*touch* anything: a headset's hand tracking was neither requested nor read, and the
only interaction the runtime had was a mouse-driven `xfrc_applied` spring
(`applyDragForces`), which needs a pointer and a screen.

Making a tracked hand interact with the simulation is three separate problems:

1. **Getting the poses.** WebXR reports 25 joints per hand, but only inside the XR
   animation frame, and only when the session asked for `hand-tracking`.
2. **Getting them into MuJoCo's frame.** The scene swizzles MuJoCo's z-up frame into
   three.js's y-up frame per body (`getPosition` / `getQuaternion`), and every
   conversion so far ran *out* of MuJoCo. A hand pose runs the other way.
3. **Making the hand a thing that collides.** A force cannot represent a hand: to
   push, block, and be leaned on, the hand has to be geometry the solver knows about.

## Decision

**One mocap body per tracked joint, spliced into the scene's MJCF at load time, plus
one inactive weld per hand for pinch grabs.**

- **Mocap bodies.** A `mocap="true"` body carries no degrees of freedom: MuJoCo
  integrates it as static geometry whose pose `mocap_pos` / `mocap_quat` set each
  step. It therefore collides like any other body while remaining perfectly
  kinematic, and — because it adds no dofs — `nq` and `nv` are untouched.
- **Injection at load, by text splice** (`core/xr/handRig.ts`). `.mjz` archives are
  unpacked into MEMFS before `mj_loadXML`, so the rig is spliced into the root XML
  before it compiles, written beside the original so relative asset paths still
  resolve. Repeated `<worldbody>` / `<equality>` sections are legal MJCF and merge on
  compile, so the splice appends rather than rewrites; ids of existing entities do
  not shift. A model the splice cannot compile falls back to the model as authored.
- **11 joints per hand** — five fingertips, four proximal knuckles, wrist and palm.
  Enough to push with a palm and pinch with any finger, at 22 spheres rather than 50.
- **Pinch grabs by weld, retargeted at runtime.** Each hand compiles one
  `<weld active="false">` from its palm to the world; on a pinch the runtime points
  `eq_obj2id` at the grabbed body, writes the relative pose the two are already in,
  and sets `eq_active`. Releasing clears `eq_active`. A weld holds an object rigidly
  with nothing to tune, where mocap contact alone cannot: a body with no velocity
  gives the friction solver nothing to grip with, so pinched objects slip out.
- **Contacts follow tracking.** An untracked hand is parked and its collisions are
  switched off, so a hand the runtime has lost is not a phantom obstacle.
- **Substep interpolation.** Poses are latched once per control step and written per
  physics substep (`stepPhysics`'s `onSubstep`), because 20 ms of hand motion applied
  as a single teleport tunnels through thin geometry.
- **An XR rig owns the camera.** three.js reads the camera's *parent* transform as
  the reference-space origin, so the camera and both hands hang off one `xrRig`
  group. On session start it is placed where the desktop camera stood, facing the
  same target; `local-floor` puts the headset's floor on MuJoCo's z=0.

### What the wasm build actually does

Three behaviours were measured against the bundled MuJoCo build rather than assumed;
each cost a bug on the way in, and each is now covered by `e2e/xr-hand.spec.ts`.

- **`eq_data` for a weld is `[anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]`**,
  and **its relpose is body2 expressed in body1's frame** — the opposite of the
  reading that a docs skim suggests. Writing the other direction snaps a grabbed
  object across the room instead of holding it still.
- **Broadphase prunes on the per-body collision aggregate** (`body_contype` /
  `body_conaffinity`), which the compiler derives from the geoms. Gating a hand's
  contacts by writing `geom_contype` alone leaves the body pruned and the hand
  ghostly: both levels have to be written.
- **`mj_resetData` does not restore `mjModel`.** Collision gating lives in the model,
  so a reset has to re-disable it explicitly or a parked hand keeps its contacts.

## Consequences

- A scene can opt out (`ViewerConfig(hand_tracking=False)`), and a browser without
  WebXR is never injected into at all.
- The rig costs 22 static bodies and 22 spheres in every injected model. They are
  compiled in `group="3"`, which the scene builder already skips
  (`geom_group[g] < 3`), so nothing is added to the three.js scene and the hands are
  drawn by three.js's own primitive hand model instead — physics and rendering split
  cleanly, with no double image.
- Because the hands are real geometry, they can touch a *robot*, not just props. A
  policy will react to being pushed, and a broadly-scoped `illegal_contact`
  termination can fire on a hand's touch. That is the feature working as intended,
  not a regression to suppress; a task that must be immune should exclude the hand
  geoms from the contact sensors it terminates on.
- Grabs are rigid. There is no compliance to feel through a weld, and no force
  feedback to give: an object welded to a mocap hand can be pushed through a wall,
  since the constraint outranks the contact. A softer *virtual coupling* (a dynamic
  hand proxy sprung to the mocap target) was considered and deferred — it needs
  per-scene tuning to earn the extra realism.
- The mouse drag path is untouched. Both are viewer disturbances, and the two never
  run at once in practice, but `applyDragForces` still owns `xfrc_applied`
  exclusively — a future force-based XR interaction would have to share it.
