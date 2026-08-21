---
icon: octicons/eye-16
---

# VR & Hand Tracking

Any built mjswan app that a headset can reach gets an **Enter VR** button. On a headset
with hand tracking — a Quest 3, for instance — your hands become part of the physics:
you can push a box over, bat a ball away, lean on a robot, or pinch something and pick
it up.

Nothing to configure. The button appears when the browser reports an `immersive-vr`
session, and hands appear when the headset reports them.

## What actually happens

Your hands enter the simulation as **mocap bodies** — one sphere per tracked joint,
spliced into the scene's model when it loads. A mocap body carries no degrees of
freedom, so MuJoCo teleports it wherever the tracker says while still solving its
contacts. That is what makes a hand something the simulation can be pushed by, rather
than a force applied at a point.

- **11 joints per hand** are tracked: five fingertips, four knuckles, the wrist and the
  palm.
- **Pinch to grab.** Bring thumb and index finger together near an object and it is
  welded to your palm until you let go. Anything with a free joint can be picked up;
  a bolted-down body cannot.
- **Lose tracking and the hands leave.** A hand the headset stops reporting is parked
  with its contacts switched off, so it never becomes an invisible obstacle.

The hands are a viewer interaction, exactly like dragging with the mouse — they are not
part of the policy's world model. The policy has no observation of your hands; it only
feels what they do. One consequence worth knowing: a task that terminates on
"illegal contact" can be ended by your own touch.

## Where you stand

When the session starts you are placed where the desktop camera was, looking at the same
point. Floor height comes from the headset's own `local-floor` space, so a standing
viewer stands on the scene's `z = 0`.

Recentre from the headset itself (long-press the Meta button on a Quest) if you would
rather face another way. Body-following is off in VR on purpose: moving the world
underneath a standing viewer is what makes people motion-sick.

## Turning it off

Hand tracking injects 22 small collision spheres into the model. They are invisible and
inert until a session starts, but if you would rather ship a scene without them:

```python
project.add_scene(spec=spec, name="G1").set_viewer(
    mjswan.ViewerConfig(hand_tracking=False)
)
```

VR itself still works — you just look rather than touch.

!!! tip "Embedded in another page?"
    An iframe needs `allow="xr-spatial-tracking"` before the viewer can enter VR at all.
    See [Embedding](embedding.md).

## Controllers

Controllers are not wired to the physics — the hand rig is driven by hand tracking only.
A headset without hand tracking (or with it switched off in system settings) still
renders the scene in VR.

## Design notes

The reasoning behind mocap bodies, the pinch weld, and what the MuJoCo WASM build
actually does with equality constraints is recorded in
[ADR 0006](https://github.com/ttktjmt/mjswan/blob/main/docs/adr/0006-vr-hand-tracking-as-mocap-bodies.md).
