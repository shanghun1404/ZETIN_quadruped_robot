"""NumPy repeated-candidate CEM qualification and distribution updates.

The caller evaluates np.repeat(unique_samples, repeats, axis=0), so replicas
of each candidate are consecutive. Selection uses the observed minimum score,
and every observed replica must pass the survival/progress/cycle gates.
Observed minima and repeated trials are not a statistical certificate of
robustness. Independent heldout evaluations remain required. No historical
best score is accepted here; callers must compare qualified incumbents using
their CURRENT generation's measurements rather than a lucky old score.
"""
from __future__ import annotations

import math
import numpy as np


def _finite_array(value, name):
    try:
        original=np.asarray(value)
        if original.dtype.kind=="c":
            raise ValueError(f"{name} must be real")
        array=np.asarray(value,dtype=float)
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _binary_array(value, name):
    array=_finite_array(value,name)
    if not ((array==0)|(array==1)).all():
        raise ValueError(f"{name} must be Boolean or binary")
    return array.astype(bool)


def aggregate(scores, alive, progress, completed_steps, target_distance, repeats):
    """Return (observed minimum scores, qualified mask, diagnostics).

    Four inputs are equal-length nonempty vectors, in consecutive replica
    groups. repeats is a positive integer dividing their length. target_distance
    is a finite nonnegative scalar [m]. Step counts must be nonnegative integers.
    Qualification requires ALL replicas: alive, progress>=.25*target_distance,
    completed_steps>=4. Boundary progress and four completed steps are inclusive.

    Unqualified groups still return their actual finite minimum, for truthful
    diagnostics. That score is NOT selectable: update gates on qualified, and
    callers must use the same mask for incumbent retention/comparison. q25 is
    the actual linearly interpolated sample quantile, never the selection score.
    Inputs are not modified. This function provides no inference about trials
    outside this observed batch or their statistical independence.
    """
    if (not isinstance(repeats,(int,np.integer)) or isinstance(repeats,(bool,np.bool_))
            or repeats<=0):
        raise ValueError("repeats must be a positive integer")
    scores=_finite_array(scores,"scores")
    alive=_binary_array(alive,"alive")
    progress=_finite_array(progress,"progress")
    completed=_finite_array(completed_steps,"completed_steps")
    distance=_finite_array(target_distance,"target_distance")
    if distance.ndim!=0 or distance<0:
        raise ValueError("target_distance must be a nonnegative scalar")
    if scores.ndim!=1 or scores.size==0 or scores.size%repeats:
        raise ValueError("scores must be a nonempty vector divisible by repeats")
    if any(value.shape!=scores.shape for value in (alive,progress,completed)):
        raise ValueError("alive, progress and completed_steps must match the score vector")
    if (completed<0).any() or (completed!=np.floor(completed)).any():
        raise ValueError("completed_steps must be nonnegative integers")
    count=scores.size//repeats
    scores,alive,progress,completed=(value.reshape(count,repeats)
                                      for value in (scores,alive,progress,completed))
    score_min=scores.min(axis=1)
    progress_min=progress.min(axis=1)
    completed_min=completed.min(axis=1)
    all_alive=alive.all(axis=1)
    progress_ok=progress_min>=.25*float(distance)
    cycle_ok=completed_min>=4
    qualified=all_alive&progress_ok&cycle_ok
    diagnostics=dict(unique_count=int(count),replicas_per_candidate=int(repeats),
        qualified_count=int(qualified.sum()),alive_count=alive.sum(axis=1),
        all_alive=all_alive,progress_min=progress_min,completed_steps_min=completed_min,
        progress_ok=progress_ok,cycle_ok=cycle_ok,score_min=score_min.copy(),
        score_q25=np.quantile(scores,.25,axis=1,method="linear"))
    return score_min,qualified,diagnostics


def update(samples, aggregate_scores, qualified, lower, upper, oldmean, oldstd):
    """Return (mean, std, details) for the NEXT full sampled generation.

    samples is (unique_candidates,parameters); scores/qualified are vectors of
    unique_candidates. Bounds and old distribution are parameter vectors.
    All real inputs must be finite, samples/oldmean within strict lower<upper
    bounds, oldstd positive. Scores for unqualified candidates never rank elites.

    With >=2 qualifiers: elite_count=min(qualified_count,max(2,ceil(.2*uniqueN))),
    select by descending observed minimum score, deterministic original-index
    tie order. Blend .3*old+.7*elite for mean and population std (ddof=0), with
    std floor .05*(upper-lower). There is no upper std cap in this branch.

    With <2 qualifiers: keep mean, expand std by 1.25 with
    cap=max(oldstd,.2*(upper-lower)). A previously wider std remains unchanged;
    failure never shrinks it. No stop/partial-population recommendation is made:
    the caller should generate and evaluate the next complete population.
    Inputs are not modified. Qualification is an observed-batch gate, not a
    statistical certificate; independent heldout evaluation remains required.
    """
    samples=_finite_array(samples,"samples")
    scores=_finite_array(aggregate_scores,"aggregate_scores")
    qualified=_binary_array(qualified,"qualified")
    lower,upper,mean,std=(_finite_array(value,name) for value,name in
        ((lower,"lower"),(upper,"upper"),(oldmean,"oldmean"),(oldstd,"oldstd")))
    if samples.ndim!=2 or not all(samples.shape):
        raise ValueError("samples must be a nonempty (unique_candidates,parameters) array")
    unique_n,dimension=samples.shape
    if scores.shape!=(unique_n,) or qualified.shape!=(unique_n,):
        raise ValueError("aggregate_scores and qualified must match unique_candidates")
    if any(value.shape!=(dimension,) for value in (lower,upper,mean,std)):
        raise ValueError("bounds and old distribution must be parameter vectors")
    with np.errstate(over="ignore"):
        span=upper-lower
    if not np.isfinite(span).all() or not (span>0).all():
        raise ValueError("lower and upper must define finite positive parameter ranges")
    if ((samples<lower)|(samples>upper)).any() or ((mean<lower)|(mean>upper)).any():
        raise ValueError("samples and oldmean must be within bounds")
    if not (std>0).all():
        raise ValueError("oldstd must be positive")
    qualified_indices=np.flatnonzero(qualified)
    qualified_count=qualified_indices.size
    if qualified_count>=2:
        elite_count=min(qualified_count,max(2,math.ceil(.2*unique_n)))
        order=np.argsort(-scores[qualified_indices],kind="stable")
        elite_indices=qualified_indices[order[:elite_count]]
        elite=samples[elite_indices]
        # Work within [0,1] to avoid unnecessary sum/squared-deviation overflow
        # for finite but large parameter units; this is the same mean/population std.
        normalized=(elite-lower)/span
        elite_mean=lower+normalized.mean(axis=0)*span
        elite_std=normalized.std(axis=0,ddof=0)*span
        # The convex mean is mathematically within bounds. Clip roundoff such
        # as .3*.04+.7*.04<.04 so the next strict input check remains valid.
        next_mean=np.clip(.3*mean+.7*elite_mean,lower,upper)
        next_std=np.maximum(.3*std+.7*elite_std,.05*span)
    else:
        elite_count=0
        elite_indices=np.empty(0,dtype=int)
        next_mean=mean.copy()
        cap=np.maximum(std,.2*span)
        with np.errstate(over="ignore"):
            next_std=np.minimum(std*1.25,cap)
    if not np.isfinite(next_mean).all() or not np.isfinite(next_std).all():
        raise ValueError("Distribution update produced nonfinite values")
    details=dict(unique_count=int(unique_n),qualified_count=int(qualified_count),
        elite_count=int(elite_count),elite_indices=elite_indices,
        updated_from_qualified=bool(qualified_count>=2))
    return next_mean,next_std,details
