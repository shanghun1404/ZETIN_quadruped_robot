# PhysX 521 ↔ source MJCF comparison — CPU job 526

## Execution and retained evidence

HPC CPU Slurm job **526** completed with `0:0` in one second, using compute/admin, two CPUs and 2 GiB; no GPU was requested. The input is the completed nominal snapshot from probe 521, not a locally generated simulation. All model calculations ran in this Slurm job. Independent coordinate algebra tests had already passed six cases locally before this run.

Input directory: `/home/shanghun1404/v16_12dof_model/physics_jobs/521`.

Local copies retain the raw `physx_comparison_521_nominal.npz`, probe manifest, Jacobian origin experiment, computed `physx_comparison_526.json` and `.npz`, all 54 body rows as CSV, and Slurm stdout/stderr/accounting. Full file digests are in `physx_comparison_526_evidence_manifest.json`; the JSON comparison itself records the input and production-code digests.

| Input/code | SHA-256 |
| --- | --- |
| Source MJCF | `8ef5712d8604110d1efc7ed92845bd83107e0ef42e1fa0f379448387c4793c6f` |
| V14 USD top-level file | `864f4b11fc24af1099acf292ce5358192294426928387c6d903cba41d45050ad` |
| Probe NPZ | `592628934257fac8856485172102c669eaa7fd5aa0bfbdcc68246286aea438dd` |
| Probe manifest | `480a80d60bffb110ef0afb544351690fbb2ca187936b05db3bcd5ccac8e76e94` |
| Comparison code | `d580efdffb3b5ef1560f8a454e5f74c7bee0f6c4af32dd0e3b608bd87722c223` |
| Source-model code | `baaddfb7067578fab156b865d19d27ae7232a357703c69e9a12c5ac7d9a0fec9` |

The USD digest is the parent probe's top-level asset digest, not a hash of its entire referenced dependency tree.

## State and coordinate conditions

This is one **stand-probe state with nonzero measured velocities**, not an exact stationary equilibrium or the derivation's ideal `q_HFE=-0.55, q_KFE=1.10` state. HFE positions span approximately −0.51628 to −0.50500 rad and KFE positions 1.16488 to 1.21292 rad. The greatest measured absolute joint speed is 0.0486389 rad/s. Root actor world linear velocity is approximately `[0.00017877, 0.02900921, 0.00892676]` m/s and angular velocity `[-0.01110037,-0.000122857,-0.00124990]` rad/s. Exact state arrays are retained in the input NPZ and computed NPZ.

All twelve joints are explicitly mapped to canonical FL, FR, RL, RR / HAA, HFE, KFE order. The source root actor pose is assigned the measured PhysX root actor pose; that actor correspondence is declared, not estimated by fitting dynamics. World gravity is `[0,0,-9.81]`. The environment origin is already present in the world actor positions and is not subtracted.

The installed Tensor SDK is 107.3.26. Its articulation inertia tensor is referenced at the link COM, expressed in the rigid-body-prim axes and packed column-major. The COM quaternion is used to check COM pose; it does not rotate the reported inertia a second time. Contract source: [NVIDIA Tensor SDK 107.3 API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html).

PhysX generalized root velocity is world COM linear/angular twist. Pinocchio uses actor-local linear/angular twist. With `r = p_COM-p_actor`, `T[:6,:6] = [[R,-skew(r)R],[0,R]]` and identity joint block, the comparison uses:

```
v_PhysX = T v_Pin
M_Pin = T.T M_PhysX T
h_Pin = T.T (h_PhysX + M_PhysX Tdot_v)
Tdot_v[:3] = R (omega × v_actor + omega × (omega × r_actor_COM))
J_Pin = J_PhysX T
```

Body and foot velocity/Jacobian comparisons evaluate the same measured PhysX world point in the source frame. A raw same-local-coordinate Jacobian comparison is also retained separately.

## Measured residuals

The following are maximum absolute component differences, with relative Frobenius norm given where useful. Different entries have their corresponding physical units; generalized mass/bias mix translation, rotation and joint coordinates.

| Quantity | Maximum absolute difference | Relative Frobenius |
| --- | ---: | ---: |
| Raw exported M vs source rigid M, 18×18 | 3.7242919002e−6 | 1.3316068604e−7 |
| Raw exported M vs source M with .02 joint armature | 2.0000291543e−2 | 8.7655752484e−4 |
| Rigid joint M block, 12×12 | 2.9154320919e−7 | 1.0881896655e−6 |
| Gravity | 4.1301180951e−5 | 1.1957943006e−7 |
| Coriolis with coordinate transport | 3.7029881366e−9 | 3.4450382161e−7 |
| Bias with coordinate transport | 4.1302101799e−5 | 1.1009235846e−7 |
| Bias if coordinate transport is omitted | 1.4739651778e−2 | 3.4641567462e−5 |
| Exported h vs exported g+c | 8.7502412498e−6 | 1.9626662982e−8 |
| Four foot COM Jacobians, 4×6×18 | 6.3575870635e−7 | 2.9969406743e−7 |

These small residuals support agreement of the **rigid-body** tensors at this state, within the reported deviations. Exact identity is not claimed. The nominal source model's `.02` armature-inclusive mass is a distinct mode.

## Root and Jacobian basis checks

The measured root actor-to-COM twist relation has maximum absolute error **4.3643708092e−10**. Source predicted actor/world and COM/world twists match the 54 measured body twists to **3.8227764849e−8** and **3.9815446324e−8**, respectively.

The probe's independent joint-velocity experiment selected foot **COM** Jacobians: COM velocity error 1.4901161e−8 m/s, versus actor-origin error 0.2407158 m/s. The comparison additionally checks all four root/link origin hypotheses using the analytic world-base Jacobian block `[[I,-skew(p_link-p_root)],[0,I]]`:

| Root world basis → evaluated link point | Maximum absolute base-column difference |
| --- | ---: |
| Root COM → link COM | 2.6999744396e−7 |
| Root COM → link actor | 6.1656263471e−1 |
| Root actor → link COM | 6.4482664684e−3 |
| Root actor → link actor | 6.2301090360e−1 |

Together these measured checks support the root COM world basis and foot COM origin used in the transform. The code reports each hypothesis; it does not silently choose a basis by the smallest residual.

## Mass, actor poses, COM and inertia

All **54 source bodies and 54 PhysX bodies** match by name, with no unmatched body. The thirteen movable-joint clusters are independently aggregated with the parallel-axis theorem, so fixed members are not counted twice.

| Quantity | Maximum difference |
| --- | ---: |
| Total mass: source 45.451322469636764 kg; PhysX 45.45132318139076 kg | 7.11753998e−7 kg |
| Body actor position | 5.45767514e−7 m (`RL_calf`) |
| Body actor rotation matrix element | 3.71959740e−7 (`FL_calf_profile`) |
| Body COM world position | 5.35009554e−7 m (`RL_calf`) |
| Individual inertia world matrix element | 1.74892882e−8 kg·m² (`body_left`) |
| Cluster mass | 8.25881955e−7 kg (`root`) |
| Cluster COM world position | 2.96676152e−7 m (`FL_HFE`) |
| Cluster inertia at its COM, world axes | 5.08669845e−8 kg·m² (`root`) |

The source total and PhysX total above are float64 sums of the stored individual masses. The probe manifest's float32 aggregate (45.45132064819336 kg) is retained as supplied; it is not substituted for the individual-mass sum. Reconstructed COM positions and rotations agree with independently exported COM pose tensors to 1.7473391e−7 m and 3.2581789e−7 per rotation-matrix component.

## Armature, damping, friction and use limits

- Measured PhysX armature is `0.019999999552965164` for every joint; source armature is `.02`. Because inclusion in the exported mass was initially unknown, raw mass is compared with **both** source rigid and source armature-inclusive modes. No armature was automatically subtracted from PhysX M.
- The exported mass agrees with the rigid mode and differs from the armature-inclusive mode by approximately `.02` on each joint diagonal. This is evidence that **this exported tensor omits rotor armature**. It does not establish whether the PhysX acceleration solver includes armature; that requires a separate force/acceleration experiment.
- Measured PhysX joint damping is zero, while source MJCF damping is `1.5`. Source damping is removed from the primary bias comparison, and its contribution is retained separately in the NPZ. The source derivation's damping-dependent state-space/discrete matrices therefore do not directly represent this configured PhysX damping mode.
- PhysX joint friction is `0.30000001192092896` for all joints. The parent tensor compensation contract excludes joint friction, damping and contact forces; this comparison has the same scope. The report does not compare contact or acceleration solver responses.
- A root actor reauthoring would require an independently measured actor gauge transform. The observed body pose errors support the declared actor correspondence at this configuration, while the explicit assumption remains recorded.
- One stand-probe configuration and velocity do not establish global MJCF/USD equivalence, gait stability or accuracy away from this state. The evidence supports using source **rigid-body** kinematics/dynamics as an independent check at this state, with armature/damping/friction modes stated separately.

Preparation code had six passing independent algebra tests and a review with no remaining Critical/Important under the declared root actor correspondence after the same-world-point correction. The subsequent read-only review of the actual job 526 report also found no remaining Critical/Important. It independently checked code hash, all body/cluster mappings, the root/foot coordinate hypotheses, the bias transport sign, and the conditional armature interpretation. It reiterated that source damping-dependent digital poles cannot be presented as the configured PhysX digital poles. No additional physics calculation or file edits were performed by the reviewer.
