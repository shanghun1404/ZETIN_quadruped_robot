"""Exact held-input discretization and all SISO channels of V16 local dynamics.

Run the CLI on the HPC. Algebra helpers have no Pinocchio dependency and can
also verify an independently specified small state-space system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def exact_zoh(A, B, period, drift=None):
    """Return Ad,Bd,dd from one augmented exponential, also for singular A."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    n, m = B.shape
    if A.shape != (n, n) or period <= 0 or not np.isfinite(period):
        raise ValueError("Inconsistent state dimensions or nonpositive period")
    drift = np.zeros(n) if drift is None else np.asarray(drift, float)
    if drift.shape != (n,) or not all(np.isfinite(item).all() for item in (A, B, drift)):
        raise ValueError("Nonfinite or inconsistent state-space data")
    augmented = np.zeros((n + m + 1, n + m + 1))
    augmented[:n, :n] = A
    augmented[:n, n:n+m] = B
    augmented[:n, -1] = drift
    propagated = expm(augmented * period)
    return propagated[:n, :n], propagated[:n, n:n+m], propagated[:n, -1]


def sampled_response(Ad, Bd, inputs, initial_state, drift=None):
    inputs = np.asarray(inputs, float)
    state = np.asarray(initial_state, float).copy()
    drift = np.zeros_like(state) if drift is None else np.asarray(drift, float)
    trajectory = [state.copy()]
    for held_input in inputs:
        state = Ad @ state + Bd @ held_input + drift
        trajectory.append(state.copy())
    return np.asarray(trajectory)


def _reachable(edges, seeds):
    result = set(int(index) for index in seeds)
    frontier = list(result)
    while frontier:
        node = frontier.pop()
        for other in np.flatnonzero(edges[:, node]):
            other = int(other)
            if other not in result:
                result.add(other)
                frontier.append(other)
    return result


def siso_channels(Ad, Bd, C):
    """Compute rational B polynomials after exact structural state pruning.

    Fixed base robot legs have disjoint six-state blocks. Removing unreachable
    or unobservable structural states avoids a badly conditioned degree-24
    denominator. This does not remove coupling within a leg. Coefficients are
    ascending powers of B: numerator starts with zero for the held-input delay.
    """
    Ad, Bd, C = (np.asarray(value, float) for value in (Ad, Bd, C))
    n = Ad.shape[0]
    if Ad.shape != (n, n) or Bd.shape[0] != n or C.shape[1] != n:
        raise ValueError("Inconsistent SISO state dimensions")
    edges = Ad != 0
    channels = []
    for output in range(C.shape[0]):
        row = []
        observable = _reachable(edges.T, np.flatnonzero(C[output]))
        for input_index in range(Bd.shape[1]):
            reachable = _reachable(edges, np.flatnonzero(Bd[:, input_index]))
            indices = sorted(reachable.intersection(observable))
            if not indices:
                row.append(dict(numerator_B=[0.], denominator_B=[1.], state_indices=[], realization="structural_zero"))
                continue
            block = Ad[np.ix_(indices, indices)]
            b = Bd[indices, input_index]
            c = C[output, indices]
            denominator = np.real_if_close(np.poly(block)).real
            numerator = np.zeros(len(indices) + 1)
            markov = []
            vector = b.copy()
            for power in range(len(indices)):
                markov.append(c @ vector)
                vector = block @ vector
                numerator[power + 1] = sum(denominator[lag] * markov[power-lag] for lag in range(power + 1))
            poles, vectors = np.linalg.eig(block)
            conditioning = float(np.linalg.cond(vectors))
            channel = dict(numerator_B=numerator.tolist(), denominator_B=denominator.tolist(), state_indices=indices, eigenvector_condition_number=conditioning)
            if conditioning < 1e8:
                residues = (c @ vectors) * np.linalg.solve(vectors, b)
                channel.update(realization="polynomial_and_modal", poles_complex_pairs=np.column_stack([poles.real, poles.imag]).tolist(), residues_complex_pairs=np.column_stack([residues.real, residues.imag]).tolist())
            else:
                channel.update(realization="polynomial_and_reduced_statespace", reduced_Ad=block.tolist(), reduced_Bd=b.tolist(), reduced_C=c.tolist())
            row.append(channel)
        channels.append(row)
    return channels


def evaluate_channels(channels, delay, *, realization="stable"):
    def evaluate(channel):
        if realization == "stable" and "poles_complex_pairs" in channel:
            poles = np.array([complex(*pair) for pair in channel["poles_complex_pairs"]])
            residues = np.array([complex(*pair) for pair in channel["residues_complex_pairs"]])
            return np.sum(residues * delay / (1-poles*delay))
        if realization == "stable" and "reduced_Ad" in channel:
            block = np.asarray(channel["reduced_Ad"])
            return np.asarray(channel["reduced_C"]) @ np.linalg.solve(np.eye(len(block))-block*delay, np.asarray(channel["reduced_Bd"])*delay)
        return np.polynomial.polynomial.polyval(delay, channel["numerator_B"]) / np.polynomial.polynomial.polyval(delay, channel["denominator_B"])
    return np.array([[evaluate(channel) for channel in row] for row in channels])


def difference_response(channel, inputs):
    """Direct scalar recurrence for zero initial state; includes input delay.

    Full output adds every channel plus the separately propagated initial state
    and affine drift. The exported state-space realization is numerically safer
    for control, particularly at small T and for near-unit repeated poles.
    """
    numerator = np.asarray(channel["numerator_B"], float)
    denominator = np.asarray(channel["denominator_B"], float)
    inputs = np.asarray(inputs, float)
    outputs = np.zeros_like(inputs)
    for sample in range(len(inputs)):
        forcing = sum(numerator[lag] * inputs[sample-lag] for lag in range(min(sample+1, len(numerator))))
        history = sum(denominator[lag] * outputs[sample-lag] for lag in range(1, min(sample+1, len(denominator))))
        outputs[sample] = (forcing-history) / denominator[0]
    return outputs


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def derive(mjcf_path, output, periods=(.005, .02)):
    from v16_physical_model import JOINT_NAMES, NOMINAL_STAND, build_model
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fixed, free = build_model(mjcf_path), build_model(mjcf_path, floating_base=True)
    q0 = NOMINAL_STAND.copy()
    M0, M_rigid = fixed.mass_matrix(q0), fixed.mass_matrix(q0, include_armature=False)
    g0 = fixed.gravity(q0)
    eps = 1e-6
    Kg = np.column_stack([(fixed.gravity(q0 + np.eye(12)[j]*eps) - fixed.gravity(q0 - np.eye(12)[j]*eps)) / (2*eps) for j in range(12)])
    D = np.diag(fixed.source_damping)
    A = np.block([[np.zeros((12, 12)), np.eye(12)], [-np.linalg.solve(M0, Kg), -np.linalg.solve(M0, D)]])
    B = np.vstack([np.zeros((12, 12)), np.linalg.solve(M0, np.eye(12))])
    C = np.hstack([np.eye(12), np.zeros((12, 12))])
    # Input u=tau-g0. q0 is a fixed-base equilibrium, so nominal affine drift is zero.
    d = np.zeros(24)
    arrays = dict(q0=q0, M0=M0, M_rigid=M_rigid, g0=g0, tau0=g0, Kg=Kg, D_source=D, armature_source=fixed.source_armature, A=A, B=B, C=C, affine_drift=d, floating_M0=free.mass_matrix(q0), floating_g0=free.gravity(q0), S=free.actuation_matrix)
    rng = np.random.default_rng(160929)
    checks = {}
    for tag, physical in (("fixed", fixed), ("floating", free)):
        M = physical.mass_matrix(q0)
        velocity = rng.normal(0, .2, physical.model.nv)
        acceleration = rng.normal(0, .3, physical.model.nv)
        torque = physical.inverse_dynamics(q0, velocity, acceleration)
        recovered = physical.forward_dynamics(q0, velocity, torque)
        raw_torque = physical.inverse_dynamics(q0, velocity, acceleration, include_armature=False, include_damping=False)
        raw_recovered = physical.forward_dynamics(q0, velocity, raw_torque, include_armature=False, include_damping=False)
        jac_error = 0.
        for leg in ("FL", "FR", "RL", "RR"):
            point = np.array([.01, -.02, .03])
            J = physical.point_jacobian(q0, leg+"_foot", point)
            finite = np.column_stack([(physical.point_after_integration(q0, np.eye(physical.model.nv)[j]*eps, leg+"_foot", point) - physical.point_after_integration(q0, -np.eye(physical.model.nv)[j]*eps, leg+"_foot", point))/(2*eps) for j in range(physical.model.nv)])
            jac_error = max(jac_error, float(np.max(np.abs(J-finite))))
            arrays[tag+"_"+leg+"_foot_origin_world"] = physical.frame_position(q0, leg+"_foot")
            arrays[tag+"_"+leg+"_foot_point_J_world"] = J
        checks[tag] = dict(mass_symmetry_max_abs=float(np.max(np.abs(M-M.T))), mass_eigenvalue_min=float(np.linalg.eigvalsh(M).min()), mass_eigenvalue_max=float(np.linalg.eigvalsh(M).max()), mass_condition_number=float(np.linalg.cond(M)), raw_rnea_aba_max_abs=float(np.max(np.abs(raw_recovered-acceleration))), armature_rnea_mass_solve_max_abs=float(np.max(np.abs(recovered-acceleration))), foot_point_jacobian_fd_max_abs=jac_error, total_mass_error=float(abs(physical.total_mass-physical.source_total_mass)))
    pin_q = fixed.configuration(q0)
    fixed.pin.computeRNEADerivatives(fixed.model, fixed.data, pin_q, np.zeros(12), np.zeros(12))
    analytic_Kg = fixed.data.dtau_dq[np.ix_(fixed.canonical_v_indices, fixed.canonical_v_indices)].copy()
    arrays["Kg_analytic"] = analytic_Kg
    checks["gravity_derivative_fd_vs_analytic_max_abs"] = float(np.max(np.abs(Kg-analytic_Kg)))
    checks["gravity_derivative_symmetry_max_abs"] = float(np.max(np.abs(Kg-Kg.T)))
    channels_by_period = {}
    discretization = {}
    equation_lines = ["State difference model: x[k+1]=Ad*x[k]+Bd*u[k]+dd; u=tau-tau0.", "x[0] is retained. y[k]=C*Ad^k*x[0]+sum_j y_ij[k]+affine response.", "Each scalar channel recurrence below starts with zero channel history.", "Polynomial arrays are ascending powers of the backshift B."]
    for period in periods:
        key = f"T{period:.3f}"
        Ad, Bd, dd = exact_zoh(A, B, period, d)
        arrays[key+"_Ad"], arrays[key+"_Bd"], arrays[key+"_dd"] = Ad, Bd, dd
        channels = siso_channels(Ad, Bd, C)
        channels_by_period[key] = channels
        freq_max_abs, freq_max_relative, polynomial_max_abs, polynomial_max_relative = 0., 0., 0., 0.
        # Frequency away from B=1 where cancellation is intrinsically severe.
        for omega in np.linspace(.02, np.pi-.02, 90):
            delay = np.exp(-1j*omega)
            reference = C @ np.linalg.solve(np.eye(24)-Ad*delay, Bd*delay)
            rational = evaluate_channels(channels, delay)
            error = np.max(np.abs(reference-rational))
            freq_max_abs = max(freq_max_abs, float(error))
            freq_max_relative = max(freq_max_relative, float(error/max(np.max(np.abs(reference)), 1e-14)))
            polynomial = evaluate_channels(channels, delay, realization="polynomial")
            poly_error = np.max(np.abs(reference-polynomial))
            polynomial_max_abs = max(polynomial_max_abs, float(poly_error))
            polynomial_max_relative = max(polynomial_max_relative, float(poly_error/max(np.max(np.abs(reference)),1e-14)))
        x0 = rng.normal(0, .01, 24)
        u0 = rng.normal(0, .05, 12)
        endpoint = solve_ivp(lambda _, x: A@x+B@u0+d, (0, period), x0, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
        zoh_endpoint = Ad@x0+Bd@u0+dd
        inputs = np.tile(u0, (100, 1))
        trajectory = sampled_response(Ad, Bd, inputs, x0, dd)
        initial_only = sampled_response(Ad, Bd, np.zeros_like(inputs), x0, dd)
        zero_initial = sampled_response(Ad, Bd, inputs, np.zeros(24), dd)
        scalar_output = np.column_stack([sum(difference_response(channels[i][j], inputs[:, j]) for j in range(12)) for i in range(12)])
        state_output = zero_initial[:-1] @ C.T
        arrays[key+"_example_inputs"] = inputs
        arrays[key+"_example_nonzero_x0"] = x0
        arrays[key+"_example_trajectory"] = trajectory
        arrays[key+"_initial_condition_trajectory"] = initial_only
        discretization[key] = dict(period_s=period, ode_vs_zoh_endpoint_max_abs=float(np.max(np.abs(endpoint-zoh_endpoint))), siso_vs_statespace_frequency_max_abs=freq_max_abs, siso_vs_statespace_frequency_max_relative=freq_max_relative, scalar_difference_vs_statespace_response_max_abs=float(np.max(np.abs(scalar_output-state_output))), initial_condition_superposition_max_abs=float(np.max(np.abs(trajectory-initial_only-zero_initial))), max_siso_denominator_degree=max(len(channel["denominator_B"])-1 for row in channels for channel in row), continuous_eigenvalues_real_range=[float(np.linalg.eigvals(A).real.min()), float(np.linalg.eigvals(A).real.max())], discrete_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(Ad)))))
        discretization[key].update(polynomial_frequency_max_abs=polynomial_max_abs, polynomial_frequency_max_relative=polynomial_max_relative)
        arrays[key+"_scalar_difference_output"] = scalar_output
        for i, row in enumerate(channels):
            for j, channel in enumerate(row):
                a, b = channel["denominator_B"], channel["numerator_B"]
                equation_lines.append(f"{key} y[{JOINT_NAMES[i]}<-{JOINT_NAMES[j]}]: denominator_B={a}; numerator_B={b}")
                rhs = " ".join(f"{-a[lag]:+.17g}*y_ij[k-{lag}]" for lag in range(1,len(a))) + " " + " ".join(f"{b[lag]:+.17g}*u_j[k-{lag}]" for lag in range(len(b)) if b[lag] != 0)
                equation_lines.append("  y_ij[k] = " + (rhs.strip() or "0"))
    # Linearization error is recorded, not interpreted as global gait validity.
    local_error = {}
    for amplitude in (.001, .01, .05):
        maximum = 0.
        for _ in range(30):
            delta = rng.uniform(-amplitude, amplitude, 24)
            torque = rng.uniform(-.05, .05, 12)
            nonlinear = np.r_[delta[12:], fixed.forward_dynamics(q0+delta[:12], delta[12:], g0+torque)]
            maximum = max(maximum, float(np.max(np.abs(nonlinear-(A@delta+B@torque)))))
        local_error[str(amplitude)] = dict(max_joint_position_rad=amplitude, max_joint_velocity_rad_s=amplitude, sample_count=30, torque_perturbation_max_Nm=.05, derivative_error_max_abs=maximum)
    sources = {str(Path(__file__).name): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "v16_physical_model.py": hashlib.sha256(Path(__file__).with_name("v16_physical_model.py").read_bytes()).hexdigest()}
    report = dict(schema_version=1, slurm_job_id=os.environ.get("SLURM_JOB_ID"), fixed_manifest=fixed.manifest(), floating_manifest=free.manifest(), joint_names=list(JOINT_NAMES), nominal=q0.tolist(), tau0_Nm=g0.tolist(), checks=checks, discretization=discretization, linearization_samples=local_error, code_sha256=sources, limitations=["Source MJCF rigid-body model, not yet matched to USD/PhysX runtime.", "Fixed-base local linearization excludes contact, floating-base motion and changing contact phase.", "Source damping and armature included explicitly; Coulomb friction/contact constraints are not simulated.", "Foot Jacobians refer to specified body-local test points, not measured mesh/PhysX contact patches.", "SISO channels retain cross coupling and require superposition over every input.", "Direct high-order polynomial evaluation near B=1 may be ill-conditioned; use exported state-space arrays for numerical control."], array_file="v16_model_tensors.npz", channels_file="v16_model_siso_channels.json")
    np.savez_compressed(output/"v16_model_tensors.npz", **arrays)
    (output/"v16_model_manifest.json").write_text(json.dumps(report, indent=2, default=_json_default)+"\n")
    (output/"v16_model_siso_channels.json").write_text(json.dumps(channels_by_period, indent=2)+"\n")
    (output/"v16_model_difference_equations.txt").write_text("\n".join(equation_lines)+"\n")
    summary = dict(job_id=report["slurm_job_id"], source_sha256=fixed.source_sha256, total_mass_kg=fixed.total_mass, checks=checks, discretization=discretization, linearization_samples=local_error)
    print(json.dumps(summary, indent=2, default=_json_default))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    derive(args.mjcf, args.output)
