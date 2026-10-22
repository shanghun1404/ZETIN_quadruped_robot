# V15 CAD Fine-tuning Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the integrated training run. Independent action and objective modules use the parallel agent workflow. No automatic commits.

**Goal:** Fine-tune adopted V15 on the unchanged CAD USD and measure improvement through the existing ROS2 acceptance suite.

**Architecture:** Keep the actor235→12 interface and frozen V12 depth student. Replace training-side joint mapping and command sampling with the CAD/ROS2 contract, retaining implicit PD and the original physical asset. Resume PPO into an isolated Slurm job snapshot; validate candidates in external ROS2 processes.

**Tech Stack:** Isaac Lab/Isaac Sim5.1, PyTorch/RSL-RL PPO, ROS2 Jazzy, Slurm compute/admin/shard.

**Spec:** Existing `docs/v15_design.md`, user-authorized CAD fine-tuning, and fixed15case acceptance `reports/v15/usd_20261011/full_suite_20261011/summary.json`.

## Global Constraints

- Adopted actor339/model11699 SHAe2bb747acb4737f9e73dc34446cfae692ca2564ed36560fb1f137814eea46d33; original checkpoint remains unchanged.
- CAD source SHA2a41b7f8ad3822b7929ec94a0399dc5cbcdd2e95e309c348d562013b9ba7df78:23.094498 kg,13 bodies,12 joints,67 original convexHull colliders.
- Physics200Hz, policy50Hz, action scale.2, LPF25Hz, both ROS2 torque guards.99, nominal gains300/8, caps24.8/48Nm.
- Keep named/sign mapping, COM versus actor-origin velocity,235observation ordering, and original V12 depth normalization.
- GPU physics and PPO only under SSHws1 Slurmcompute/QoSadmin, gres/shard:1. Never alter other jobs.
- No hardware deployment or replacement of the adopted checkpoint until existing acceptance criteria pass. Training rewards never change acceptance thresholds.
- Source snapshots and failed candidates are retained. Job-local sources isolate this experiment from the existing dirty workspace.

## Review Focus

- Mirrored joints and native articulation order: compare against actual ROS2 filter/motor command.
- Zero/reverse/lateral/yaw commands: preserve requested signed direction and yaw; no heading override.
- Self-collision versus foot support: use66 filtered nonadjacent body pairs, not ground net forces.
- Curriculum cheating: side bypass or termination must not promote; four actual risers determine progress.
- Depth peers: one robot per generated terrain column; preserve camera path and frozen student.

## Tasks

- [x] Add CAD action/sign/LPF/double-guard adapters and random-state ROS2 parity tests (`config/v15_cad_training_actions.py`, tests13).
- [x] Add balanced commands and rated/self-contact costs (`config/v15_cad_training_mdp.py`, meaningfulCPU tests).
- [x] Assemble seven actual stair heights2/3.5/5/7.5/10/12.5/15.7cm; fixed4risers/.3m steps/2m width. Verify ground changes and promotion exclusions (`config/v15_cad_training_terrain.py`, tests4).
- [x] Integrate `GuideV15CadEnvCfg` and `train_v15_cad.py`; job2330 verified23.09449768kg/67colliders/235obs and100physical control steps, zero cleaned nonfinite events and zero physics-divergence termination. Native RSL-RL TensorDict and hidden-divergence regressions:6passed. Failed pre-PPO job2329 is archived.
- [x] Resume339 optimizer, fixedLR5e-5 in both LR owners, initialstdboost.18 (subsequently trainable), seed73. Job2330 completed128envs/2000updates and6,144,000PPOsamples on50%flat/25%up/25%down. Finalmodel13699 SHA1a3ac12414ffbcb69ed68245f6df90e3dd7971258c4bf576ccc434fa7d7b4357; all parameters/scalar logs finite, nonfinite/divergence counters0, original339 unchanged. Actor deltaL2=5.3758826; finalstd.0739225–.1294663. All intermediate checkpoints remain onws1.
- [x] Evaluate selected checkpoints using the unchanged ROS2 flat/stop/reverse/lateral/yaw/stair/rough/standing/integrated sequence cases. Record per-joint phase current estimates and ratedRMS separately from contact reactions.
- [x] Compare to baseline15results; report actual success/failure and retain rejected candidates. Add holdout speed/stop seeds only if needed to distinguish measured improvement from a threshold fluctuation.

## Verification

CPU: existing114tests plus explicit new action/objective/terrain tests; no recursive pytest over archived job sources. Slurm outputs must show unchanged CAD mass/collider count, finite235observations, preflight PASS, actual resumed LR, and saved finite checkpoint. Physics adoption uses existing suite task_success and trace checks, preserving all original failures.

## Decisions

User explicitly authorized fine-tuning after reviewing the15case failures. Continue implementation and training without another permission request. No commits because workspaceAGENTS forbids automatic commits. A curriculum promotes only after actual traversal and keeps the acceptance suite independent of training rewards.

## Running Jobs

- Training2330:128CAD environments,2000PPOupdates; original339actor/optimizer, fixedLR5e-5, initialstdboost.18.
- Evaluation2331:9flat/command/rough/standing/integrated cases, dependencyafterok:2330.
- Evaluation2332:6stair cases, dependencyafterok:2330.
- Both evaluation jobs require training_completed.json, strictpreflightPASS and exact final-checkpoint SHA; seed42 and existingcasecriteria are retained. Results will be archived under `reports/v15/cad_finetune_20261011`.

## Round 1 measured outcome and bounded follow-up

- Job2330 completed2000updates; external ROS2 jobs2331/2332 completed all15cases with unchanged criteria. Candidate7/15 versus baseline2/15: flat, stop_restart, 5cmup, 10cmdown, standing, .10/.18m sit-rise-walk-stop-sit pass. Reports and180motor-comparison rows are archived.
- Remaining failures: reverse/lateral/yaw collapse toward rest, rough stalls before tiles, 5cmdown corridor drift, 10cmup incomplete, 15.7cmup falls with1345.328N self-contact, 15.7cmdown stops after one riser. All failures remain recorded; no adoption.
- Follow-up limited to1000PPOupdates from2330final,128environments/fixedLR5e-5/seed73/initialstdboost.25. Targeted command probabilities zero/forward/reverse/lateral/yaw/combined=.10/.10/.30/.25/.20/.05 on nonforced flat environments, 8–14s segments, clipped signed progress weight4. Planarstd.20/yawstd.10. Rated-load costs remain; lane/self-contact penalties strengthened.
- Training tiles40%flat/10%rough/25%up/25%down; rough seed73 versus unchanged holdout seed42. Geometry is assembled from the same box contract; rough surface-height observations match authored boxes. No mass, collider, joint/cap, camera student or ROS2 evaluation changes.
- Corridor drift from a remaining absolute yaw offset is observable in the global suite, while actor235 observations contain body velocity and no absolute lane heading. Training penalties cannot guarantee recovery; retain strict corridor failures.
- Follow-up requires strict100physical-step preflight, finite checkpoint and full15case external ROS2 validation before reporting its outcome.

Follow-up submitted as training2333; unchanged-suite evaluation2334(stairs)/2335(other9), afterok2333. Native CPU55tests pass; independent review found no critical issue. Actual128column allocation52flat/12rough/32up/32down.

Follow-up training2333 completed1000updates/3,072,000samples (total3000updates/9,216,000samples across both stages), Slurmelapsed31:03; finalmodel14699 SHA b04417c29f30e47020195b070d059984a37531ed163c65b237c05b90539da5df. All1000nonfinite/divergence counters0; all scalar values and saved policy parameters finite. Actor delta versus2330 L2=4.4740801; std.0741655–.1613074. Original339 unchanged,286snapshotfiles hashverified. Evaluation2334/2335 running undercompute/admin with unchanged seed42.

## Final measured outcome

All15unchanged seed42 ROS2/Isaac cases completed for both candidates; trace/transport checks15/15, all source hashes verified. Baseline2/15 → first7/15 → second12/15. Second passes flat, stop/restart, reverse, lateral, rough, 5cmup/down,10cmup/down, standing, .10/.18m sit-rise-walk-stop-sit. Yaw and15.7cmup/down remain failed. All15second-candidate recorded self-contact maxima0N; this does not imply fall prevention or real readiness.

Reverse RR_KFE RMS19.19194Nm>rated18; yaw FR_HAA8.419995Nm>8.3;15.7cmup10joints and15.7cmdown1joint exceed rated RMS. Estimated drive peaks remain below24.8/48Nm, while contact reactions are reported separately. Flat phase RMS3.223–10.262A is higher at the maximum than first2.415–7.357A and baseline3.344–8.093A; all second-candidate flat joints remain within rated RMS. No universal current reduction claim. All four5/10cm stair cases have0ratedRMSexceed joints.

Final candidate model14699 SHA b04417c29f30e47020195b070d059984a37531ed163c65b237c05b90539da5df; all checkpoint variants retained, original339 unchanged.24asset comparisons and39ROS2files match preserved snapshots; physics criteria unchanged. Reports include actual recorded durations (shortened on early failure), separate yawrad/s plot axis,180per-joint comparison rows per stage and15videos per stage. Overview `reports/v15/cad_finetune_20261011/overview.html`. Candidate retained for evaluation; no real deployment or automatic adoption.

## User-requested non-stair RMS optimization

User excluded stairs for now and authorized optimization of reverse RR_KFE19.19/18Nm and yaw FR_HAA8.42/8.3Nm. Goal: all nine unchanged non-stair motion cases, self-contact and drive checks pass, and zero rated RMS exceedances across all12motors.

Resume2333final for600updates/128envs/fixedLR3e-5/initialstdboost.16/seed73; flat116/rough12 training columns, no stairs. Command probabilities zero/forward/reverse/lateral/yaw/combined=.10/.20/.25/.10/.30/.05, hold14–18s, directed-progress6/yawtracking8. Add training-only200Hz squared-torque EWMA with.5s timeconstant, rewardsum(max(rms/(.90*rated)-1,0)^2) weight-12; ratedmeanweight-.75 and original instantaneous-excess cost retained. Physical target/guard/LPF/gain interfaces unchanged. True evaluation RMS remains over each full recorded case, not this training surrogate.

TDD and nativeCPU checks for EWMA pulses/resets/constant-load and exact native terrain masks; strict100step Isaac preflight including authored rough-height query. Freeze eachjob source. Evaluate fixedseed42 cases in two externalROS2 jobs: flat/stop/reverse/lateral/yaw and rough/standing/two sit-rise-walk-stop-sit sequences. Retain2333model and previous reports; no auto deployment, no commits. Reports underreports/v15/cad_rms_20261011.

RMS follow-up submitted as training2338 (600updates/128envs, compute/admin/shard1). Evaluation2339(flat/stop/reverse/lateral/yaw) and2340(rough/standing/two sit-rise sequences) depend on successful2338. NativeCPU97checks pass; independent review found no submission blocker. Rough-height assertion checks analytical ground lookup against authored boxes, not measured PhysX geometry. All9unchanged non-stair full-record RMS and motion gates remain required.

RMS2338 completed600updates/1,843,200samples, all600healthlogsfinite/nonfinite/divergence0, actorchangedL2=1.9100643. Finalmodel15299 SHA07712abc8c261926744b57608e719603c35c5f18fc62228fac7130126ec6ec19. ROS2jobs2339/2340 7/9motionpass; allrecordedcaseRMSwithinrated, reverseRR_KFE19.19194→13.34064Nm andyawFR_HAA8.419995→3.875972Nm, yawtrackingnowpasses. Bothsit-riseintegrationcasesFAIL afteractor→WBC handover, preservevideos/fulltimelines. Firstfall19.525s/contact1527.9299N; deepcasecontrollerlandingIKfailure19.06s is secondary tosupportcollapse.

Root-cause evidence: same17.0BRAKE/17.3BLEND_STAND/18.3STABILIZEtiming. Old2333 recoveredfourcontacts18.02s. New2338 retainsFL/RRtwolegsupport after18.3policyweight0, missingFR/RLtouchdown, lateralCOMescape/fall. Actualhullgaps~.2–2mm exceednominalmaterialpointapproxerror~.1mm. NoROS2mapping/transportfailure evidence. Native training never performsactor→WBCblend.

Boundedremedy: resume2338for400updates, addfrozen2333teacher actor-meanMSEweight5onlyforforward/zero commandmask(abs(vx)<=.02,vy>=0,abs(yaw)<=.02) intoeachnativePPOminibatchbeforebackward. PreserveRMSprofile/geometry/rewards/commands/ROS2controller/evalcriteria, so reverse/yaw improvements remain free tooptimize. Runtimepolicyarchitecture/checkpointplain235→12 unchanged. Independent meaningfulCPUgradient/nativeequivalence tests and review precedeSlurmsubmission. Repeatall9fixedseed42cases withfullRMS/contact/drive/tracegates. Nooriginalcheckpointorfailedrecordoverwrites.

Retentionfollow-upsubmittedtraining2341(400updates128envs3e-5,.10initialstdminimum);externalROS2suite2342(flat/stop/reverse/lateral/yaw)and2343(rough/standing/twosit-rise)afterok2341. NativeCPU113pass3.76s; independentreviewnoblockingdefect. CustomPPOstaticupdateusesreviewedrsl-rl-lib3.1.2sourceSHA4373ac1b2f9fdf14d9da57516968fc95d8f605d2967fee01dc61bf0d09423478(BSD-3-Clause),teacherfrozenMSEweight5innativebackward.

## Non-stair RMS optimization final evidence (2026-10-12 KST)

Job2341 completed400updates/1,228,800samples, Slurm14:54/0:0. All400 health counters zero, scalar/checkpoint parameters finite. Actor deltaL2=1.4060085; teacher mean-squared loss first50mean.00509574→last50mean.00146490. Finalmodel15699 SHA9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6. RMS optimization total600+400=1000updates/3,072,000samples; original339 and both previous candidates unchanged.

ExternalROS2 jobs2342(11:52)/2343(15:26), compute/admin, bothCOMPLETED0:0. All nine original fixedseed42 non-stair cases pass at their full14/24/12/30second durations, including both10/18cm sit-rise-walk-stop-sit sequences. Zero rated RMS exceedances across108case/joint records, allnonadjacentcontactmaxima0N, boundedprestepPD drivepeakswithin24.8/48Nm, allnine200HztracechecksPASS. Source287files/jobverified; frozenphysicaltrees4assets/39ROS2/7CADmodel files byte-identical to2333; criteria/protocol unchanged.

Reverse RR_KFE RMS19.191942→12.499897Nm (−34.8690%, estimatedphaseRMS8.680484A). Yaw FR_HAA8.419995→4.156988Nm (−50.6296%, estimatedphaseRMS3.379665A), yaw motion nowpasses. Flat estimatedphaseRMS3.261390–9.937319A; no universalcurrentreductionclaim. FirstfailedRMS candidate remainsarchived. Finalreportreports/v15/cad_rms_20261011/round2/report.html,108motorcomparisons,9videos/fulltimelines, source/trace/checkpoint/trainhealthverification. Simulation scope met; stairs excluded, no hardwaredeployment or automaticcommit.
