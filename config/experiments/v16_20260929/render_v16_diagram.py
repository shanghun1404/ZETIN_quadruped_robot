"""Create standalone V16 vector/raster diagrams; no training/simulation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parents[3] / "reports/v16/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def box(ax, x, y, w, h, label, color="#e8f1fa", fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=color, edgecolor="#35516b", linewidth=1.3))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs)


def arrow(ax, start, end, color="#35516b", style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                color=color, linewidth=1.3, linestyle=style))


def setup(title):
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set(xlim=(0, 15), ylim=(0, 8.5))
    ax.axis("off")
    ax.text(.3, 8.1, title, fontsize=20, weight="bold", color="#203447")
    return fig, ax


physical, ax = setup("V16 | 12 actuated joints + a floating base")
box(ax, 4.8, 6.25, 5.4, 1.15, "Floating base: position + orientation\n6 unactuated velocity coordinates", "#d9e8f5", 14)
for x, leg in zip((1.2, 4.4, 7.6, 10.8), ("FL", "FR", "RL", "RR")):
    arrow(ax, (7.5, 6.25), (x+1.1, 5.8))
    for y, joint, limit in ((4.95, "HAA", "24.8"), (3.55, "HFE", "48.0"), (2.15, "KFE", "48.0")):
        box(ax, x, y, 2.2, .8, f"{leg}_{joint}\npeak limit {limit} N m")
        if y > 2.2:
            arrow(ax, (x+1.1, y), (x+1.1, y-.6))
    arrow(ax, (x+1.1, 2.15), (x+1.1, 1.65))
    box(ax, x, .95, 2.2, .65, f"{leg} foot contact", "#e6f1e5")
ax.text(.5, .3, "54 source rigid members: mass / COM / inertia aggregated without dropping fixed members.\n"
        "M(Q) vdot + h(Q,v) = S^T tau + Jc(Q)^T lambda.  v: 18 dimensions; torque input: 12.", fontsize=12)

control, ax = setup("V16 | digital model-based control and parameter learning")
nodes = [(.4, 5.8, 2.45, 1.15, "Prepared crawl + COM\nminimum-jerk waypoints"),
         (3.3, 5.8, 2.45, 1.15, "Foot COM Jacobian IK\nbody + foot references"),
         (6.2, 5.8, 2.45, 1.15, "Digital feedback\njoint + body acceleration"),
         (9.1, 5.8, 2.5, 1.15, "Force QP / whole-body QP\ncontact + joint torque"),
         (12.0, 5.8, 2.5, 1.15, "Torque/speed/slew guard\n20 ms command hold"),
         (12., 3.55, 2.5, 1.15, "PhysX nonlinear plant\n5 ms physics integration"),
         (8.6, 3.55, 2.7, 1.15, "State + ground contacts\npresence / loaded / wrench"),
         (5.2, 3.55, 2.7, 1.15, "18 x 18 model tensors\nM + armature / h / J"),
         (1.2, 3.55, 3.2, 1.15, "CEM parameter learning\nrepeated walking evaluation")]
for x,y,w,h,label in nodes:
    box(ax,x,y,w,h,label, "#e7f2e6" if "PhysX" in label else "#e8f1fa")
for a,b in zip(nodes[:4],nodes[1:5]):
    arrow(ax,(a[0]+a[2],6.38),(b[0],6.38))
arrow(ax,(13.25,5.8),(13.25,4.7))
arrow(ax,(12.,4.12),(11.3,4.12))
arrow(ax,(8.6,4.12),(7.9,4.12))
arrow(ax,(6.55,4.7),(7.4,5.8))
arrow(ax,(7.9,4.7),(10.3,5.8))
arrow(ax,(8.6,3.8),(4.4,3.8))
arrow(ax,(2.8,4.7),(1.7,5.8),style="--")
box(ax, .6, 1.65, 13.8, 1.0, "GUIDELINE: physical laws -> ODE -> Laplace + initial conditions -> latched step\n"
    "-> continuous response -> sampled coupled difference -> B operator -> MIMO -> SISO channels -> recurrence", "#f4eee0", 12)
ax.text(.65,.55,"12 torque inputs; no virtual base actuator. Force-only QP fixes acceleration; whole-body QP can adjust it.\n"
    "Both log finite-iteration constraint violations. COM reference bounds do not bound actual robot motion.\n"
    "Safety projection can change torque within a hold interval. Linear/SISO derivation is local, not a global walking proof.\n"
    "Learning tunes controller parameters; an independent seed evaluation is required before claiming success.",fontsize=11)

with PdfPages(OUT / "v16_modeling_diagrams_architecture.pdf") as pdf:
    for fig,name in ((physical,"v16_physical_model"),(control,"v16_controller_architecture")):
        fig.savefig(OUT/f"{name}.svg",bbox_inches="tight")
        fig.savefig(OUT/f"{name}.png",dpi=180,bbox_inches="tight")
        pdf.savefig(fig,bbox_inches="tight")
        plt.close(fig)
print(OUT)
