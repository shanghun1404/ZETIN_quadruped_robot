# V16 independent holdout evaluation

Execution completion and walking quality are reported separately.

| Run | Survived | Speed (m/s) | Progress (m) | Power (W/joint) | Gates |
|---|---:|---:|---:|---:|---|
| model_seed1001 | 32/32 | 0.0537 | 0.5881 | 0.766 | True |
| model_seed1002 | 32/32 | 0.0541 | 0.5920 | 0.770 | True |
| model_seed1003 | 32/32 | 0.0536 | 0.5865 | 0.764 | True |
| pd_default_seed1001 | 0/32 | -0.0124 | -0.2049 | 0.426 | False |

Model holdout quality passed: **True**.

95% survival, speed error <=0.05 m/s and progress 75–125% of commanded distance are required per seed.
Torque data are explicit commands. Solver contact/friction forces and hardware thermal limits are separate.
