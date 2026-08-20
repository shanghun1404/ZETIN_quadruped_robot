# Porting the 45 kg MJCF quadruped into IsaacLab, and the first training run on it

*Drafted by GPT-5.6-sol (xhigh) from measured data; reviewed and corrected. Two
corrections applied are marked **[corrected]**.*

## Summary

The MuJoCo `quadruped.xml` robot was ported into the IsaacLab project and
completed its first flat-ground training run.

The port required loading the MJCF importer explicitly, reducing the converted
asset to one articulation root, deactivating an embedded world plane, applying
the new per-joint torque limits, and replacing collider-based fall termination
with orientation-based termination.

The converted asset has incorrect per-body frame positions. This broke
body-position diagnostics, the stock height scanner, generated-terrain spawning,
and the interpretation of the forward axis. Root rotation, velocity, tilt and
contact-force measurements remain valid.

V10-flat completed 3000 iterations in 43 minutes. The best checkpoint was
`model_2000.pt`: speed 0.503 m/s against a 0.5 m/s command, survival 1.00, foot
contact fractions between 0.494 and 0.527. However the robot's actual forward
axis is base **+y** while V10-flat commanded base x, so the behaviour is a
sideways crab walk, not a fore-aft trot.

## The new robot

| Property | v1 | v2 |
|---|---:|---:|
| Source asset | `guide_quadruped.urdf` | `quadruped.xml` MJCF |
| Links / bodies | 18 | 54 |
| Total mass | 25.34 kg | 45.451 kg, PhysX-confirmed |
| Trunk link | `base_link` | `base` |
| Joints | 12: FL/FR/RL/RR x HAA/HFE/KFE | same names |
| HAA torque limit | 53.0 N m | 24.8 N m |
| HFE/KFE torque limit | 53.0 N m | 48.0 N m |
| Stand keyframe | not provided | base z 0.4640 m; each leg (0, -0.55, 1.10) |

Chassis, 20.000 kg over six bodies: `base` 5.345, `body_right` 4.285,
`body_left` 4.285, `body_head` 2.698, `body_front` 1.693, `body_back` 1.693.

Legs, 25.451 kg over 48 bodies = 6.363 kg per leg: thigh 1.443, HFE motor 0.960,
calf 0.960, thigh cover 0.894, knee bracket 0.799, HAA motor 0.521, hip 0.254,
compro1 0.152, compro2 0.152, profile 0.099 x2, foot 0.030.

v1 represented each leg with four links totalling 3.21 kg. v2 separately
represents motors, brackets, covers and profiles. The mass difference is a
modelling-fidelity difference, not a different machine size.

## Asset pipeline

### 1. Load the MJCF importer

- **Symptom** `MjcfConverter` fails on Isaac Sim 5.1 with
  `AttributeError: 'NoneType' object has no attribute 'set_import_sites'`.
- **Cause** `MJCFCreateImportConfig` is unregistered; the importer extension is
  not loaded by default.
- **Fix** `enable_extension("isaacsim.asset.importer.mjcf")` before conversion.
- **Evidence** Conversion succeeds once the extension is enabled.

### 2. Reduce the asset to one articulation root

- **Symptom** `Failed to find a single articulation`.
- **Cause** `PhysicsArticulationRootAPI` on both `/quadruped/base/base` and
  `/quadruped/worldBody`.
- **Fix** Keep the robot root, remove the API from the wrapper.
- **Evidence** Inspection found exactly those two roots.

### 3. Inspect instance proxies before diagnosing missing collision geometry

- **Symptom** First inspection reported no robot collision geometry:
  `num_colliders == 1`, that one the world plane.
- **Cause** Every body's `visuals`/`collisions` scope is instanced, and
  `Stage.Traverse()` does not descend into instances — it reported zero meshes.
- **Fix** Traverse with `Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())`.
- **Evidence** Proxy-aware traversal finds five colliders: four `*_foot` plus the
  plane. That matches the MJCF, which sets `<default><geom contype="0"
  conaffinity="0"/></default>` and re-enables collision on the four
  `*_foot_geom` and the plane. **The collision conversion was correct all along.**

### 4. Remove the embedded world plane from the spawned asset

- **Symptom** The robot stood on a floor it carried with it and never fell.
- **Cause** The MJCF world plane converted to
  `/quadruped/worldBody/_/collisions/CollisionPlane`, a sibling of the robot
  root, so the spawn transform moved it with the robot. Per environment that is
  4096 infinite planes.
- **Fix** Author a strong root-layer opinion: `SetActive(False)` on
  `/quadruped/worldBody`.
- **Constraint** Do **not** use `RemovePrim`. The asset is a USDC stub composed
  from subLayers, references and a payload; `RemovePrim` removes only local
  opinions and does not reliably suppress a composed prim.

### 5. Do not compensate with chassis and hip bounding boxes

- **Symptom** Bounding-box colliders added to `base` and the four `*_hip` so that
  contact termination could fire reduced ground support from 99.4% to 56%, with
  FL and RL at 0 N.
- **Cause** The boxes were sized from bounds carrying the conversion's baked
  offsets and struck the ground, tilting the robot and lifting two feet.
- **Fix** Reverted. Use orientation-based termination instead.

## Scrambled body frames, and the four failures they caused

Per-body frames are displaced from their geometry, by a different amount per
body. One constant offset cannot correct it. Standing robot, feet in the base
frame:

| Foot | position [x, y, z] m |
|---|---|
| FL | [-0.347, -0.208, -0.196] |
| FR | [+0.351, +0.283, **+0.190**] |
| RL | [-0.347, -0.843, -0.202] |
| RR | [-0.011, -0.844, -0.182] |

The FR foot frame reads 0.190 m *above* the trunk while the robot is standing.

The root frame's **rotation** is valid — rest tilt 1.3 degrees, projected gravity
clean — so velocity, tilt and contact-force metrics are usable. Per-body
positions are not.

**1. A false "the robot fell through the floor".** Diagnosed from `body_pos_w`.
Contact force showed 443.2 N against 445.9 N own weight (99.4%): it was standing.

**2. A blind height scan.** `mdp.height_scan` returns
`sensor.data.pos_w.z - hit_z - offset`, measured from the parent body frame. With
the stock offset 0.5 the scan sat near -0.92. Recalibrating the offset to the
settled frame value (-0.4192) still left it 96.9% pinned at the -1 clip
(mean -0.9688, std 0.1741). Replacing the observation with `median(hit_z) - hit_z`
— a difference between two ray hits, with no body frame in it — gave saturation
0.0%, mean 0.0003, range -0.239 to +0.290.

**3. Invalid terrain spawning.** `init_state.pos` is a frame value. Setting the
frame to -0.389 (intended to put the feet 0.03 m above the tile origin) placed
the robot under the mesh: zero contact force, root z reaching -314 m, 100%
`base_tilt` terminations, mean episode length 8 of 1000. Spawning at +2.0 made
contact (99.4% support) but the robot landed and flipped to 166 degrees; +0.6
flipped to 165.6. **Generated terrain is unresolved.**

**4. A misidentified forward axis.** Below.

## The forward-axis error

Two independent measurements on a standing robot:

```
front_pair - rear_pair  = [0.181, 0.881, 0.191]     0.88 m apart in y
left_pair  - right_pair = [-0.517, -0.245, -0.202]  0.52 m apart in x
whole-robot extent xyz  = [0.80, 1.42, 0.58]        longest horizontal axis is y
```

The fore-aft axis is **y** and the head points **+y**. The v1 robot faced base
-x, and that convention was inherited without rechecking it against the new
asset. The old robot's own config even carries the comment *"this robot's
left/right axis is Y, and forward is +X"* — exactly inverted for v2.

So V10-flat commanded velocity along base x and the robot learned to travel
**sideways** at the commanded speed.

Why no metric caught it: forward speed was measured on the same wrong axis, and
survival, duty factor and contact fractions are direction-agnostic. The only gate
that reacted was the FL-RR diagonal phase correlation, -0.020 against a >= 0 bar.
**The error was visible in the video before it was visible in the numbers.**

## V10-flat results

Configuration: the project's V8.1 objective (published IsaacLab Spot locomotion
reward table plus a contact-duty floor at 0.35, weight -3.0), flat ground, no
height scanner, no camera, plain MLP ActorCritic, 48-dimensional proprioceptive
observation, hidden 512/256/128, ELU, action scale 0.2, 50 Hz control on 200 Hz
physics, 4096 environments, 3000 iterations at 0.87 s/iteration (43 minutes).

Two changes forced by the asset: per-joint torque limits (HAA 24.8, HFE/KFE
48.0), and `bad_orientation` at 60 degrees replacing contact termination, because
`base` and the hips own no collider so `illegal_contact` can never fire.

Training end state: mean episode length 953.6/1000, `time_out` 96.5%,
`base_tilt` 3.5%, gait reward 8.447, base linear velocity reward 4.354 of 5.0,
duty-floor penalty -0.0002.

Checkpoint sweep, commanded 0.5 m/s:

| iteration | gates | speed m/s | min foot contact | base height m | survival |
|---:|---:|---:|---:|---:|---:|
| 1000 | 8/8 | 0.541 | 0.485 | 0.425 | 1.00 |
| **2000** | **8/8** | **0.503** | **0.504** | 0.426 | **1.00** |
| 2999 | 7/8 | 0.499 | 0.494 | 0.428 | 1.00 |

Selected: `model_2000.pt`. Per-foot contact fractions FL 0.494 / FR 0.521 /
RL 0.527 / RR 0.511.

### Comparison with v1 **[corrected]**

The draft merged two different v1 checkpoints into one row. Both are given here.

| metric | v1 V8.1 @2000 (its best) | v1 V8.1 @7999 (final) | v2 V10-flat @2000 |
|---|---:|---:|---:|
| gates | 8/8 | 8/8 | 8/8 |
| speed m/s (cmd 0.5) | 0.679 | 0.599 | 0.503 |
| survival | 1.000 | 0.969 | 1.000 |
| min foot contact | 0.454 | 0.354 | 0.504 |
| contact fractions | 0.48/0.45/0.47/0.46 | 0.67/0.36/0.35/0.66 | 0.494/0.521/0.527/0.511 |
| duty-floor penalty | — | -0.187 | -0.0002 |

On v2 the duty floor never binds (-0.0002 against v1's -0.187 at its final
checkpoint) and the split is 49-53% rather than 35-67%, at roughly half the
torque-to-weight ratio. **These v2 numbers describe a sideways crab walk and do
not establish fore-aft trotting performance.**

Three runs in this project peaked before their final checkpoint: V8 better at
1600 than 8000; V8.1 best at 2000 of 8000; V10-flat best at 2000 of 3000.

## What is still open

- **Generated terrain.** Unresolved. Frame-dependent spawning put the robot below
  the mesh at -0.389 m; +0.6 and +2.0 both produced landing flips. The chassis and
  hip bounding-box colliders reduced support and were reverted. Flat ground avoids
  every frame-dependent coupling.
- **Corrected-axis training.** V10.1 is running: identical to V10-flat except the
  forward command moves from base x to base +y. No lateral penalty was added,
  deliberately, to keep it one change. No result yet.
- **Asset correction.** The clean long-term fix is a re-export with every body
  frame aligned to its geometry. That would retire most of this document.

## Reproducing the port

1. **Use the supplied v2 MJCF.** Extract `quadruped_sim_ready.zip`, use
   `quadruped.xml`. Trunk body is `base`; 12 joints keep the FL/FR/RL/RR x
   HAA/HFE/KFE names.

2. **Load the importer before converting.**
   `enable_extension("isaacsim.asset.importer.mjcf")`, before constructing
   `MjcfConverter`.

3. **Enforce one articulation root.** Keep `/quadruped/base/base`; remove
   `PhysicsArticulationRootAPI` from `/quadruped/worldBody`.

4. **Deactivate the imported world body.** `SetActive(False)` on
   `/quadruped/worldBody`, authored into the root layer. Not `RemovePrim`.

5. **Validate collisions through instance proxies.** Use
   `Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())`. Before suppressing
   `worldBody` it must find the four `*_foot` colliders plus the plane. Do not add
   bounding-box colliders to `base` or `*_hip`.

6. **Apply the v2 physical configuration.** Total mass 45.451 kg. Torque limits
   HAA 24.8, HFE/KFE 48.0 N m. Standing joint pose from the MJCF keyframe:
   HAA 0, HFE -0.55, KFE 1.10.

   **[corrected]** Do **not** use the keyframe's base z (0.4640) as
   `init_state.pos.z`. That is a MuJoCo world value, and IsaacLab's
   `init_state.pos` is a *frame* value on an asset whose frames are offset from
   its geometry. Express the spawn height as a clearance above the feet instead:
   the settled root frame reads -0.4192 on flat ground with the feet on the
   floor, so the feet sit about 0.42 m above the root frame, and
   `init_state.pos.z = desired_foot_clearance - 0.42`. Flat ground used a 0.05 m
   clearance.

7. **Replace contact-based fall termination.** `illegal_contact` cannot fire for
   `base` or the hips — they have no colliders. Use `bad_orientation` at 60
   degrees.

8. **Keep the first run on flat ground.** Height scanner and camera off. If a
   height observation is needed later, do not use the body-frame expression
   `sensor.data.pos_w.z - hit_z - offset`; use `median(hit_z) - hit_z`.

9. **Set the axes.** Forward is base **+y**; left/right is base x; the head points
   +y. Both the velocity command and the measured forward-speed metric must use
   +y. Evaluating a +y policy under the -x convention reports a forward speed of
   roughly zero.

10. **Training configuration.** V8.1 objective; Spot reward table; duty floor 0.35
    at weight -3.0; 48-dim proprioception; MLP 512/256/128 ELU; action scale 0.2;
    200 Hz physics, 50 Hz control; 4096 environments; 3000 iterations.

11. **Preflight, before spending GPU hours.** Assert all of:
    - PhysX total mass ~45.45 kg (one environment, allowing the mass-randomisation
      event's -2/+4 band).
    - Exactly one articulation root.
    - `worldBody` inactive; four foot colliders present; no chassis or hip boxes.
    - **Standing support**: ground carries 0.8-1.25 of the robot's weight, and rest
      tilt is under 20 degrees. Two runs were wasted before this check existed —
      one fell through the terrain at 0 N, one landed and flipped to 166 degrees,
      and every other check passed both.
    - **Fore-aft axis**: the measured front-minus-rear foot separation must lie on
      the same axis and sign as the commanded forward direction, and the lateral
      command band must be zero. This check would have caught the crab walk before
      training started.
    - Per-body positions are never used as geometry positions.

12. **Evaluate intermediate checkpoints.** At least 1000, 2000 and the final one,
    at the commanded speed. Do not assume the last is best — three runs in this
    project peaked earlier.
