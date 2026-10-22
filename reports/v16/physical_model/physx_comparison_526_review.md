# Read-only independent result review

Reviewer: `/root/v16_control_research`.

Reviewed `v16_compare_physx.py` and actual CPU job 526 results without further simulator/model execution or file edits.

Verdict: no remaining Critical/Important in the declared root actor correspondence and one-snapshot scope.

- Current comparison code hash matches the executed report; all 54 bodies and 13 movable-joint clusters are represented.
- Body actor/COM positions, cluster inertia and the same-world-point foot Jacobians support the alignment.
- Root COM → link COM base-column residual 2.70e−7 clearly differs from the three alternative origin hypotheses (.61656, .00644827, .62301).
- Bias error changes from .01473965 without coordinate transport to 4.13021e−5 with transport; Coriolis alone is 3.70e−9, supporting the formula and sign.
- Measured armature ≈.02 while exported mass matches rigid mode and differs by .02000029 from source armature-inclusive mode. The interpretation is limited to this snapshot/getter; solver armature effects cannot be inferred from it.
- PhysX damping is zero. The source damping 1.5 used for the fixed-base derivation must remain explicit; those source digital poles are not demonstrated as the actual configured USD poles.
