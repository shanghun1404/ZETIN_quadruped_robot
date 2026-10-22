# V16 independent holdout evaluation

Execution completion and walking quality are reported separately.

| Run | Survived | Speed (m/s) | Progress (m) | Power (W/joint) | Gates |
|---|---:|---:|---:|---:|---|
| model_seed1001 | 32/32 | 0.0498 | 1.5433 | 0.668 | True |
| model_seed1002 | 32/32 | 0.0498 | 1.5435 | 0.667 | True |
| model_seed1003 | 32/32 | 0.0498 | 1.5436 | 0.667 | True |
| pd_default_seed1001 | 0/32 | -0.0044 | -0.2049 | 0.151 | False |

Model holdout quality passed: **True**.

95% survival, speed error <=0.05 m/s and progress 75–125% of commanded distance are required per seed.
Torque data are explicit commands. Solver contact/friction forces and hardware thermal limits are separate.
