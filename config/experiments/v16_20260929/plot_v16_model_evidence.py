"""Plot HPC-produced physical matrices and trajectories; no dynamics calculation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[3]
data = np.load(root / "reports/v16/physical_model/v16_model_tensors.npz", allow_pickle=False)
out = root / "reports/v16/figures"
names = [f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("HAA", "HFE", "KFE")]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, key, labels, title in zip(axes, ("M0", "floating_M0"),
    (names, ["base_vx", "base_vy", "base_vz", "base_wx", "base_wy", "base_wz"] + names),
    ("Fixed base: 12 actuated joints", "Floating base: 6 unactuated + 12 joint velocities")):
    matrix = data[key]
    diagonal = np.sqrt(np.diag(matrix))
    coupling = matrix / diagonal[:, None] / diagonal[None, :]
    im = ax.imshow(coupling, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(title)
fig.colorbar(im, cax=fig.add_axes([.90, .25, .015, .50]),
             label="M[i,j] / sqrt(M[i,i] M[j,j]) (dimensionless)")
fig.suptitle("Source MJCF physical inertia coupling | nominal stand | computed on HPC job 520")
fig.subplots_adjust(left=.08, right=.85, bottom=.18, top=.9, wspace=.32)
for suffix in ("png", "pdf"):
    fig.savefig(out / f"v16_inertia_coupling.{suffix}", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for column, period in enumerate((.005, .020)):
    prefix = f"T{period:.3f}"
    trajectory = data[f"{prefix}_example_trajectory"]
    initial = data[f"{prefix}_initial_condition_trajectory"]
    t = np.arange(len(trajectory)) * period
    for j, name in enumerate(names[:3]):
        axes[0, column].plot(t, trajectory[:, j], label=name)
        axes[1, column].plot(t, initial[:, j], label=name)
    axes[0, column].set_title(f"T = {period*1000:g} ms, held inputs + nonzero initial state")
    axes[1, column].set_title("Initial-state contribution retained separately")
    for row in range(2):
        axes[row, column].set_xlabel("Time [s]")
        axes[row, column].set_ylabel("Joint position perturbation [rad]")
        axes[row, column].legend(fontsize=8)
fig.suptitle("12-step derived local model: saved exact-ZOH response (fixed base, no contact)")
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(out / f"v16_zoh_initial_conditions.{suffix}", dpi=160)
plt.close(fig)
