# CAD standing dynamics

`quadruped_cad.xml` is a dynamics interchange derived from the authored USD audit, for the existing Pinocchio parser. It preserves all 13 CAD bodies, their 23.09449815750122 kg total mass, COMs, full inertia tensors, and joint anchors. It contains no simulation collision geometry. Isaac loads `assets/quadruped_20261011/quadruped_robot.usda` and its original meshes.

Use `v151_ros2_control.cad_standing.CADStandingController(model_path, dt=.02)` with `model_path` pointing at `quadruped_cad.xml`; keep `cad_model.json` beside it. The inherited standing API accepts canonical FL, FR, RL, RR × HAA, HFE, KFE arrays. Native CAD leg names are FL, FR, BL, BR, with signs FL=(-1,-1,+1), FR=(-1,+1,-1), RL=(+1,-1,+1), RR=(+1,+1,-1). Position, velocity, and effort use the same signs.

The base pose and base linear velocity passed to this controller are at the base actor origin. Velocities use body axes; quaternion order is wxyz. The CAD base COM offset is [0.010427819564938545, 0.015405300073325634, 0.06333552300930023] m. The CAD bridge publishes actor-origin odometry, which the controller passes directly to WBC. It adds `cross(omega_body, r_com)` only when building the locomotion policy's COM velocity input. If a different ROS odometry source supplies the base COM, the adapter must first convert position with `p_origin = p_com - R @ r_com` and body velocity with `v_origin = v_com - cross(omega_body, r_com)`.

Four massless Pinocchio frames are attached to the actual shank bodies at the tennis meshes' zero-pose AABB centers. Their fixed material contact points are projected to the nominal ground. They add no foot bodies, inertias, or colliders. Shank rotation changes the actual mesh's lowest point, so this material-point model is a contact approximation that requires PhysX validation.

The nominal joints are [0,-0.55,1.10] rad per leg and the nominal actor-origin height is 0.4608854063842892 m. The standard crouch lowers that origin by 0.10 m to 0.36088540638428923 m; the deep crouch lowers it by 0.18 m to 0.2808854063842892 m. Both hold the four nominal material foot points. These are upright, all-feet crouches: the trunk remains elevated and does not rest on the ground. `crouch.json` supplies the exact asymmetric CAD joint values and geometry for both poses. The caller adds 0.006 m at spawn to cover the tennis mesh's small penetration relative to the material-point contact approximation.

Runtime conventions are rotor armature 0.02 kg m², no added viscous damping in WBC bias, joint friction 0.3, and torque caps [24.8,48,48] Nm per leg. WBC uses a separately defined conservative contact coefficient mu=0.3; this is not the joint friction or the USD material coefficient. The original rubber foot material remains static=0.9/dynamic=0.8, and the test ground static=0.8/dynamic=0.7. The runtime position drive uses 300 Nm/rad and 8 Nm s/rad at 200 Hz; the controller's `dt` is its own update period. `CADStandingController.friction` requires the inherited solver to pass the WBC coefficient to `solve_wbc_qp`.

`verification.json` records CPU checks only. Nominal CAD foot FK differs from the independent USD audit by at most 1.67e-16 m; inverse/forward dynamics differ by 9.55e-15; finite-difference foot Jacobians differ by 5.29e-11. Static WBC is numerically feasible at nominal stand and both crouches. The original convex hulls have no nonadjacent interior intersections at those endpoints or 13 samples over the 0.18 m sit/rise path. Directly adjacent bodies are excluded according to the USD's existing joint collision flags. These checks do not simulate PhysX mesh cooking, contact offsets, or transitions.

`collision_hulls.npz` retains convex hulls extracted from every enabled authored collider in its actual owner-body frame. A positive control folding HAA joints inward by 0.3 rad detects intersecting front and rear foot meshes, confirming the screen detects overlap rather than always returning an empty list.

CPU generation on the workstation, without SimulationApp:

```bash
PYTHONPATH=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
LD_LIBRARY_PATH=/opt/isaacsim-env/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin \
OPENBLAS_NUM_THREADS=1 /opt/isaacsim-env/bin/python config/derive_v15_cad_model.py \
  --audit reports/v15/usd_20261011/audit/usd_cpu_audit.json \
  --usd assets/quadruped_20261011/quadruped_robot.usda \
  --output config/experiments/v15_cad_20261011/model --verify --crouch-drop .10
```
