"""Render mathematical control block diagrams from saved V16 results.

This reads checkpoints and model tensors and verifies algebraic identities.
It does not run a robot model, rollout, training, or a simulator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/v16/figures"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12,
                     "mathtext.fontset": "stix", "svg.fonttype": "none"})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gains(job):
    path = ROOT / f"reports/v16/physics_jobs/{job}/checkpoint.json"
    checkpoint = json.loads(path.read_text())
    code = ROOT / f"reports/v16/source_snapshots/{job}/v16_controller_math.py"
    assert digest(code) == checkpoint["controller_code_sha256"][code.name]
    spec = importlib.util.spec_from_file_location(f"v16_math_{job}", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p, T = checkpoint["best_parameters"], checkpoint["control_dt_s"]
    kp, kd = module.digital_acceleration_gains(p[0], p[1], T)
    return dict(job=job, T=T, kp=kp, kd=kd, alpha=p[11], omega_n=p[0], zeta=p[1],
                checkpoint_sha256=digest(path), controller_math_sha256=digest(code))


def verify(gains):
    """Check transfer expressions against independent state-space resolvents."""
    records = []
    for g in gains:
        T, kp, kd, alpha = (g[key] for key in ("T", "kp", "kd", "alpha"))
        A = np.array([[1., T], [0., 1.]])
        B = np.array([T*T/2., T])
        K = np.array([(1.+alpha)*kp, kd])
        closed = A - np.outer(B, K)
        errors = []
        for angle in np.linspace(.08, 3., 61):
            z = np.exp(1j*angle)
            Gq, Gv = T*T*(z+1.)/(2.*(z-1.)**2), T/(z-1.)
            Fr = kp + kd*(1.-1./z)/T
            analytical = Gq*Fr/(1.+K[0]*Gq+kd*Gv)
            state_space = np.linalg.solve(z*np.eye(2)-closed, B*Fr)[0]
            errors.append(abs(analytical-state_space))
            Dcl = (z-1.)**2 + kd*T*(z-1.) + .5*K[0]*T*T*(z+1.)
            assert abs(Dcl-np.linalg.det(z*np.eye(2)-closed)) < 2e-12
        assert max(errors) < 2e-12
        # The gain-design pole targets apply before adding posture feedback.
        no_posture = A - np.outer(B, [kp, kd])
        desired = np.exp(np.roots([1., 2*g["zeta"]*g["omega_n"], g["omega_n"]**2])*T)
        assert np.max(np.abs(np.sort_complex(np.linalg.eigvals(no_posture))-
                             np.sort_complex(desired))) < 2e-12
        continuous = np.array([[0., 1.], [-K[0], -kd]])
        for s in (1.+2j, 3.+10j, 2.+30j):
            Frs = kp + kd*s
            expected = Frs/(s*s+kd*s+K[0])
            actual = np.linalg.solve(s*np.eye(2)-continuous, [0., Frs])[0]
            assert abs(expected-actual) < 2e-12
        records.append(dict(**g, maximum_frequency_identity_error=max(errors),
            homogeneous_feedback_poles=[[float(v.real), float(v.imag)] for v in np.linalg.eigvals(closed)],
            interpretation="ideal compensated joint axis; fixed reference input; no clipping/contact-QP modification"))
    return records


def verify_saved_model(gains):
    path=ROOT/"reports/v16/physical_model/v16_model_tensors.npz"
    with np.load(path) as saved:
        M,D,K,A,B,C=(saved[name] for name in ("M0","D_source","Kg","A","B","C"))
        initial=np.linspace(-.03,.03,24)
        transfer_errors,initial_errors=[],[]
        for s in (.7+2j,1.+5j,3.+15j):
            plant=np.linalg.inv(M*s*s+D*s+K)
            state_plant=C@np.linalg.solve(s*np.eye(24)-A,B)
            transfer_errors.append(float(np.max(np.abs(plant-state_plant))))
            initial_from_ode=plant@(M@(s*initial[:12]+initial[12:])+D@initial[:12])
            initial_from_state=C@np.linalg.solve(s*np.eye(24)-A,initial)
            initial_errors.append(float(np.max(np.abs(initial_from_ode-initial_from_state))))
        assert max(transfer_errors)<2e-10
        assert max(initial_errors)<2e-10
        virtual_errors=[]
        for g in gains:
            dc=M*g["kd"]-D
            kc=M*((1.+g["alpha"])*g["kp"])-K
            actual_a=np.block([[np.zeros((12,12)),np.eye(12)],
                               [-np.linalg.solve(M,K+kc),-np.linalg.solve(M,D+dc)]])
            ideal_a=np.block([[np.zeros((12,12)),np.eye(12)],
                              [-(1.+g["alpha"])*g["kp"]*np.eye(12),-g["kd"]*np.eye(12)]])
            virtual_errors.append(float(np.max(np.abs(actual_a-ideal_a))))
        assert max(virtual_errors)<2e-10
    return dict(tensors_sha256=digest(path),maximum_transfer_identity_error=max(transfer_errors),
                maximum_initial_condition_identity_error=max(initial_errors),
                maximum_virtual_feedback_identity_error=max(virtual_errors),
                scope="saved fixed-base source nominal matrices; no new physical model calculation")


def page(title, subtitle):
    fig, ax = plt.subplots(figsize=(17., 10.5))
    ax.set(xlim=(0., 17.), ylim=(0., 10.5), aspect="equal")
    ax.axis("off")
    ax.text(.45, 10.0, title, fontsize=21, weight="bold")
    ax.text(.45, 9.45, subtitle, fontsize=11)
    return fig, ax


def block(ax, x, y, w, h, label, size=17):
    ax.add_patch(Rectangle((x,y),w,h,facecolor="white",edgecolor="black",linewidth=1.4))
    ax.text(x+w/2.,y+h/2.,label,ha="center",va="center",fontsize=size,linespacing=1.5)


def wire(ax, points, arrow=True):
    if len(points)>2:
        ax.plot(*zip(*points[:-1]),color="black",linewidth=1.4)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",
                                   mutation_scale=13,linewidth=1.4,color="black"))
    else:
        ax.plot(*zip(*points[-2:]),color="black",linewidth=1.4)


def dot(ax,x,y):
    ax.add_patch(Circle((x,y),.047,color="black"))


def summing(ax,x,y):
    ax.add_patch(Circle((x,y),.23,facecolor="white",edgecolor="black",linewidth=1.4))
    ax.text(x,y,r"$\Sigma$",ha="center",va="center",fontsize=17)


def element(ax, x, y, w, h, label, color="#274c67", size=16):
    """A single physical operation, rather than an equation-filled container."""
    ax.add_patch(Rectangle((x,y),w,h,facecolor="white",edgecolor=color,linewidth=1.6))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=size,color=color)


def actual_wbc_blocks_page():
    """The deployed WBC path, with predicted and physical contact kept distinct."""
    fig, ax = page("V16  |  Actual WBC controller and physical loop",
        "Whole-body QP job 577. Control update 20 ms; PhysX and emergency motor envelope 5 ms. All paths below are used by the runtime.")
    ax.text(.55,8.96,"SAMPLED ACCELERATION TASK",fontsize=13,weight="bold",color="#274c67")
    joint_rows=[(8.47,r"$q_r-q$",r"$K_p$"),
                (7.80,r"$v_r-\dot q$",r"$K_d$"),
                (7.13,r"$q_0-q$",r"$\alpha K_p$")]
    ax.add_patch(Rectangle((3.62,6.77),.44,2.,facecolor="#eef3f7",edgecolor="#274c67",linewidth=1.4))
    for y,source,gain in joint_rows:
        element(ax,.53,y-.26,1.12,.52,source,size=13)
        wire(ax,[(1.65,y),(2.08,y)])
        element(ax,2.08,y-.26,1.03,.52,gain,size=16)
        wire(ax,[(3.11,y),(3.62,y)])
        ax.text(3.84,y,"+",fontsize=16,ha="center",va="center")
    wire(ax,[(4.06,7.79),(5.65,7.79)])
    ax.text(4.45,8.04,r"$a_j$",fontsize=13)
    element(ax,4.18,6.42,1.65,.61,"body task $a_b$",size=13)
    element(ax,5.65,7.4,.88,.77,"stack",size=13)
    wire(ax,[(5.83,6.72),(6.09,7.4)])
    element(ax,7.05,7.39,1.65,.79,"WBC QP\n320 ADMM",size=13)
    wire(ax,[(6.53,7.79),(7.05,7.79)])
    ax.text(6.64,8.24,r"$a_d\in\mathbb{R}^{18}$",fontsize=11)
    element(ax,6.37,6.13,2.28,.56,r"$M,h,J_c$ oracle",size=13)
    wire(ax,[(8.42,6.69),(8.42,7.39)])
    element(ax,4.29,5.76,1.56,.53,"stance / limits",size=11)
    wire(ax,[(5.85,6.02),(6.05,6.02),(6.05,6.95),(7.30,6.95),(7.30,7.39)])
    ax.text(8.84,8.96,"TORQUE FROM DYNAMICS",fontsize=13,weight="bold",color="#274c67")
    wire(ax,[(8.7,7.79),(9.14,7.79)])
    dot(ax,9.14,7.79)
    ax.text(8.75,8.05,r"$(a^*,\hat\lambda^*)$",fontsize=10)
    element(ax,9.73,8.21,1.38,.54,r"$Ma^*$",size=17)
    element(ax,9.73,7.49,1.38,.54,r"$h(Q,v)$",size=15)
    element(ax,9.73,6.77,1.38,.54,r"$J_c^T\hat\lambda^*$",size=15)
    wire(ax,[(9.14,7.79),(9.14,8.48),(9.73,8.48)])
    wire(ax,[(9.14,7.79),(9.14,7.04),(9.73,7.04)])
    ax.text(9.18,8.54,r"$a^*$",fontsize=11)
    ax.text(9.18,6.80,r"$\hat\lambda^*$",fontsize=11)
    ax.plot([8.65,9.46,9.46],[6.41,6.41,7.76],color="#758594",linewidth=1.15,linestyle="--",zorder=1)
    ax.add_patch(FancyArrowPatch((9.46,7.76),(9.73,7.76),arrowstyle="-|>",
                                 mutation_scale=12,linewidth=1.15,color="#758594"))
    ax.add_patch(Rectangle((11.53,6.62),.47,2.13,facecolor="#eef3f7",edgecolor="#274c67",linewidth=1.4))
    for y,sign in ((8.48,"+"),(7.76,"+"),(7.04,"−")):
        wire(ax,[(11.11,y),(11.53,y)])
        ax.text(11.77,y,sign,fontsize=16,ha="center",va="center")
    wire(ax,[(12.,7.76),(12.37,7.76)])
    element(ax,12.37,7.45,.83,.62,r"$S$",size=18)
    wire(ax,[(13.2,7.76),(14.3,7.76)])
    ax.text(14.43,7.76,r"$\tau^*$",fontsize=18,va="center")

    ax.text(.55,5.42,"DIGITAL ACTUATOR AND SIMULATED PLANT",fontsize=13,weight="bold",color="#274c67")
    y=4.48
    ax.text(.55,y,r"$\tau^*$",fontsize=18,va="center")
    blocks=[(1.33,1.31,r"$\Gamma_T$"),(3.22,1.4,"ZOH"),
            (5.18,1.44,r"$\Gamma_{5\rm ms}$"),(7.2,2.15,"PhysX rigid\ncontact"),
            (9.96,1.43,"sensors"),(11.95,2.0,r"$Q,v,\lambda_{\rm phys}$")]
    prev=1.04
    for x,w,label in blocks:
        wire(ax,[(prev,y),(x,y)])
        element(ax,x,y-.38,w,.76,label,size=13 if "PhysX" in label or "contact" in label else 15)
        prev=x+w
    ax.text(1.34,3.86,"20 ms slew",fontsize=10)
    ax.text(3.22,3.86,"20 ms hold",fontsize=10)
    ax.text(5.18,3.86,"5 ms speed/effort",fontsize=10)
    wire(ax,[(13.95,y),(15.37,y),(15.37,2.9),(.4,2.9),(.4,6.18),(.63,6.18)])
    element(ax,.63,5.88,2.44,.60,r"sampled state $x_k$",size=13)
    ax.plot([1.0,.25,.25],[6.48,6.48,8.47],color="black",linewidth=1.4)
    for row_y,_,_ in joint_rows:
        wire(ax,[(.25,row_y),(.53,row_y)])
    ax.add_patch(FancyArrowPatch((3.07,6.03),(4.29,6.03),arrowstyle="-|>",
                                 mutation_scale=12,linewidth=1.1,color="#758594",linestyle="--"))
    ax.plot([3.07,3.46],[6.32,6.32],color="#758594",linewidth=1.1,linestyle="--")
    ax.add_patch(FancyArrowPatch((3.46,6.32),(4.18,6.72),arrowstyle="-|>",
                                 mutation_scale=12,linewidth=1.1,color="#758594",linestyle="--"))
    ax.text(4.15,3.08,"next 20 ms: measured state and contact return to reference, dynamics oracle and WBC",fontsize=10,color="#274c67")
    ax.plot([12.95,12.95,8.65],[4.86,5.83,5.83],color="#758594",linewidth=1.1,linestyle="--")
    wire(ax,[(8.65,5.83),(8.65,6.4)])

    ax.text(.55,2.43,"QP CONSTRAINTS AND LIMITS",fontsize=13,weight="bold",color="#274c67")
    element(ax,.58,1.43,4.5,.61,r"$M_ba+h_b=J_b^T\hat\lambda$",size=16)
    element(ax,5.33,1.43,3.1,.61,r"$J_ca=a_c$",size=16)
    element(ax,8.68,1.43,3.68,.61,"friction / normal force",size=13)
    element(ax,12.61,1.43,3.68,.61,"motor torque bounds",size=13)
    ax.text(.6,.62,"Gait/IK builds q_r; the dashed model path supplies h. QP and torque reconstruction share M, h, J. Predicted force differs from physical contact.",fontsize=11)
    return fig


def physical_force_page():
    """Floating-base physical plant, decomposed into one operation per block."""
    fig, ax = page("V16  |  Physical plant built from small blocks",
        "18-velocity-coordinate PhysX plant. Physical contact force comes from the contact solver, not the controller QP prediction.")
    ax.text(.6,8.88,"EFFORT SOURCES",fontsize=14,weight="bold",color="#274c67")
    ax.text(7.45,8.88,"STATE EVOLUTION",fontsize=14,weight="bold",color="#274c67")
    lanes=[
        (8.16,r"$\tau_{\rm applied}$",r"$S^T$","+","12 motor torques"),
        (7.36,r"$\lambda_{\rm phys}$",r"$J_{\rm phys}(Q)^T$","+","ground forces"),
        (6.56,r"$Q$",r"$g(Q)$","−","gravity"),
        (5.76,r"$Q,v$",r"$c(Q,v)$","−","inertial bias"),
        (4.96,r"$d$",r"$I$","+","residual force"),
    ]
    for y,source,operator,sign,meaning in lanes:
        ax.text(.7,y,source,fontsize=17,va="center")
        wire(ax,[(1.75,y),(2.35,y)])
        element(ax,2.35,y-.29,1.65,.58,operator,size=13 if "J_" in operator else 16)
        wire(ax,[(4.,y),(6.3,y)])
        ax.text(4.2,y+.19,meaning,fontsize=10,color="#54616c")
    ax.add_patch(Rectangle((6.3,4.6),.55,3.95,facecolor="#eef3f7",edgecolor="#274c67",linewidth=1.5))
    for y,_,_,sign,_ in lanes:
        ax.text(6.58,y,sign,fontsize=18,ha="center",va="center",color="#274c67")
    ax.text(6.32,4.35,"sum of efforts",fontsize=11,color="#274c67")
    wire(ax,[(6.85,6.65),(7.45,6.65)])
    element(ax,7.45,6.22,1.5,.86,r"$M_{\rm eff}(Q)^{-1}$",size=13)
    wire(ax,[(8.95,6.65),(9.4,6.65)])
    ax.text(9.02,6.94,r"$\dot v$",fontsize=13)
    element(ax,9.4,6.22,1.07,.86,r"$\int dt$",size=19)
    wire(ax,[(10.47,6.65),(11.3,6.65)])
    ax.text(10.64,6.94,r"$v_{18}$",fontsize=13)
    element(ax,11.3,6.22,1.26,.86,r"$N(Q)$",size=18)
    wire(ax,[(12.56,6.65),(13.05,6.65)])
    ax.text(12.58,6.94,r"$\dot Q$",fontsize=13)
    element(ax,13.05,6.22,1.07,.86,r"$\int dt$",size=19)
    wire(ax,[(14.12,6.65),(15.05,6.65)])
    ax.text(15.2,6.65,r"$Q_{19}$",fontsize=17,va="center")
    ax.text(7.45,5.64,"mass matrix",fontsize=11,color="#274c67")
    ax.text(9.38,5.64,"velocity",fontsize=11,color="#274c67")
    ax.text(11.3,5.64,"quaternion map",fontsize=11,color="#274c67")
    ax.text(13.05,5.64,"position",fontsize=11,color="#274c67")
    ax.text(7.45,4.98,r"$\dot v=M_{\rm eff}^{-1}(S^T\tau+J^T\lambda-g-c+d)$",fontsize=19)

    ax.text(.6,3.55,"WHAT EACH BLOCK MEANS",fontsize=14,weight="bold",color="#274c67")
    element(ax,.65,2.2,3.27,.95,"$S^T\\tau=[0_6;\\tau_{12}]$\nno base motor",size=13)
    element(ax,4.13,2.2,3.27,.95,"$J^T\\lambda$\nfoot force to generalized effort",size=11)
    element(ax,7.61,2.2,3.27,.95,"$M_{\\rm rigid}$\n54 masses, COMs, inertias",size=13)
    element(ax,11.09,2.2,3.27,.95,"rotor armature\n$0.02\\ {\\rm kg\\,m^2}$ per joint",size=13)
    ax.text(.65,1.18,r"$M_{\rm eff}=M_{\rm rigid}+S^T(0.02I_{12})S,\quad h=c+g,\quad\dot Q=N(Q)v$",fontsize=20)
    ax.text(.65,.82,"State-dependent M, J, g, c and N read the current Q, v; the outer state return is on page 1.",fontsize=11)
    ax.text(.65,.48,"Actual USD joint damping is zero. d includes unmodeled joint friction. Contact forces can be distributed; J^T lambda is their generalized-force reduction.",fontsize=11)
    return fig


def physical_state_page():
    """Twelve-axis nominal feedback as separate gain, sum, inertia and integrator blocks."""
    fig, ax = page("V16  |  Nominal 12-joint equivalence — analysis only",
        "Fixed-base, no-contact source model; not the runtime WBC. Blue gains are physical, orange gains are ideal equivalent control terms.")
    ax.text(.58,8.95,"REFERENCE TORQUE",fontsize=14,weight="bold",color="#274c67")
    ax.text(.58,8.51,r"$r\ [{\rm rad}]$",fontsize=14,va="center")
    ax.text(.58,7.86,r"$v_r\ [{\rm rad/s}]$",fontsize=14,va="center")
    element(ax,1.8,8.22,.95,.55,r"$K_p$",size=18)
    element(ax,1.8,7.57,.95,.55,r"$K_d$",size=18)
    wire(ax,[(1.45,8.51),(1.8,8.51)])
    wire(ax,[(1.45,7.86),(1.8,7.86)])
    summing(ax,3.42,8.2)
    wire(ax,[(2.75,8.5),(3.2,8.34)])
    wire(ax,[(2.75,7.85),(3.2,8.07)])
    element(ax,4.08,7.88,1.18,.65,r"$M_0$",size=19)
    wire(ax,[(3.65,8.2),(4.08,8.2)])
    wire(ax,[(5.26,8.2),(6.27,8.2),(6.27,7.25)])
    ax.text(5.28,8.45,r"$u_{\rm ref}\ [{\rm N\,m}]$",fontsize=13)

    y=7.0
    summing(ax,6.27,y)
    ax.text(6.59,7.17,"+",fontsize=17)
    element(ax,7.2,6.6,1.28,.8,r"$M_0^{-1}$",size=19)
    wire(ax,[(6.5,y),(7.2,y)])
    wire(ax,[(8.48,y),(9.02,y)])
    ax.text(8.51,7.27,r"$\ddot y$",fontsize=13)
    element(ax,9.02,6.6,1.05,.8,r"$\int dt$",size=19)
    wire(ax,[(10.07,y),(11.64,y)])
    ax.text(10.35,7.27,r"$\dot y\ [{\rm rad/s}]$",fontsize=13)
    element(ax,11.64,6.6,1.05,.8,r"$\int dt$",size=19)
    wire(ax,[(12.69,y),(15.5,y)])
    ax.text(15.64,y,r"$y\ [{\rm rad}]$",fontsize=17,va="center")
    dot(ax,10.75,y)
    dot(ax,13.85,y)

    ax.text(.6,5.86,"FEEDBACK PATHS",fontsize=14,weight="bold",color="#274c67")
    ax.add_patch(Rectangle((5.82,2.35),.5,3.62,facecolor="#eef3f7",edgecolor="#274c67",linewidth=1.4))
    rows=[
        (5.57,r"$D_0$","source viscous damping",False),
        (4.72,r"$D_c$","controller-added damping",True),
        (3.87,r"$K_g$","gravity torque gradient",False),
        (3.02,r"$K_c$","controller-added stiffness",True),
    ]
    for row,gain,meaning,control in rows:
        ax.text(.65,row,meaning,fontsize=13,va="center",color="#a86625" if control else "#274c67")
        element(ax,7.45,row-.27,1.15,.54,gain,color="#a86625" if control else "#274c67",size=17)
        wire(ax,[(7.45,row),(6.32,row)])
        ax.text(5.97,row,"+",fontsize=16,ha="center",va="center",color="#274c67")
    wire(ax,[(10.75,y),(10.75,4.72)])
    wire(ax,[(10.75,5.57),(8.6,5.57)])
    wire(ax,[(10.75,4.72),(8.6,4.72)])
    wire(ax,[(13.85,y),(13.85,3.02)])
    wire(ax,[(13.85,3.87),(8.6,3.87)])
    wire(ax,[(13.85,3.02),(8.6,3.02)])
    wire(ax,[(5.97,5.97),(5.97,6.74),(6.1,6.86)])
    ax.text(5.55,6.72,"−",fontsize=18)
    ax.text(10.95,5.56,"velocity signal",fontsize=11,color="#54616c")
    ax.text(14.08,3.86,"position signal",fontsize=11,color="#54616c")

    ax.text(.6,2.04,"COEFFICIENTS WITH PHYSICAL UNITS",fontsize=13,weight="bold",color="#274c67")
    element(ax,.65,1.12,3.55,.68,r"$D_c=M_0K_d-D_0$",size=16)
    element(ax,4.48,1.12,3.55,.68,r"$K_c=M_0(1+\alpha)K_p-K_g$",size=15)
    element(ax,8.31,1.12,3.55,.68,r"$D_{\rm eff}=D_0+D_c$",size=16)
    element(ax,12.14,1.12,3.55,.68,r"$K_{\rm eff}=K_g+K_c$",size=16)
    ax.text(.65,.51,"D [N m s/rad], K [N m/rad]. Source D0 = 1.5 I; actual USD damping = 0. This nominal equivalence is not the runtime WBC on page 1.",fontsize=11)
    return fig


def continuous_page():
    fig, ax = page("V16  |  Continuous-time transfer-function block diagram",
        "Ideal compensated joint axis: r = q_ref - q0, y = q - q0. The sampled equivalent is on the next appendix page.")
    y = 8.1
    ax.text(.45,y,r"$R(s)$",fontsize=19,va="center")
    block(ax,2.,7.55,3.25,1.1,r"$F_r(s)=K_p+K_d s$",18)
    summing(ax,6.35,y)
    block(ax,8.,7.5,2.8,1.2,r"$G_a(s)=\frac{1}{s^2}$",23)
    wire(ax,[(1.3,y),(2.,y)])
    wire(ax,[(5.25,y),(6.12,y)])
    wire(ax,[(6.58,y),(8.,y)])
    ax.text(7.1,y+.27,r"$A(s)$",fontsize=16,ha="center")
    ax.text(5.9,y+.2,"+",fontsize=16)
    wire(ax,[(10.8,y),(15.35,y)])
    ax.text(15.55,y,r"$Y(s)$",fontsize=19,va="center")
    dot(ax,13.4,y)
    block(ax,7.5,5.7,4.6,1.0,r"$H_y(s)=(1+\alpha)K_p+K_d s$",18)
    wire(ax,[(13.4,y),(13.4,6.2),(12.1,6.2)])
    wire(ax,[(7.5,6.2),(6.35,6.2),(6.35,7.87)])
    ax.text(6.64,7.64,"−",fontsize=19)
    ax.text(.55,4.65,r"$\frac{Y(s)}{R(s)}="
            r"\frac{K_d s+K_p}{s^2+K_d s+(1+\alpha)K_p}$",fontsize=26)
    ax.text(.55,3.7,"The underlying torque-to-angle plant retains the 12-joint coupling:",fontsize=13)
    ax.text(.55,2.9,r"$G_{\tau q}(s)=\left(M_0s^2+D_0s+K_g\right)^{-1}"
            r"\in\mathbb{C}^{12\times12}$",fontsize=24)
    ax.text(.55,2.0,r"$Y(s)=G_{\tau q}(s)U(s)+Y_{\rm IC}(s),\quad "
            r"Y_{\rm IC}=G_{\tau q}\left[M_0(sy_0+v_0)+D_0y_0\right]$",fontsize=20)
    ax.text(.55,1.17,r"$G_{\tau q,d}(z)=C_q(zI-A_d)^{-1}B_d"
            r",\qquad (A_d,B_d)=\mathrm{ZOH}_{\tau,T}(A,B)$",fontsize=19)
    ax.text(.55,.25,"Upper loop: continuous-time equivalent of the joint acceleration feedback, with ideal inverse dynamics.\n"
        "Lower plant: fixed-base, no-contact source nominal model; D0 = 1.5 I. Actual USD damping is 0.\n"
        "Transfer functions describe zero-initial input response; the initial-condition source is shown separately.",fontsize=11)
    return fig


def discrete_page(gains):
    fig, ax = page("V16  |  z-domain feedback: ideal acceleration-ZOH channel",
        "T = 0.020 s. Ideal acceleration hold after compensation; actual V16 holds torque. Measured velocity is a separate feedback.")
    y = 8.1
    ax.text(.4,y,r"$R(z)$",fontsize=18,va="center")
    block(ax,1.7,7.55,3.3,1.1,r"$F_r(z)=K_p+K_d\frac{1-z^{-1}}{T}$",18)
    summing(ax,6.25,y)
    block(ax,8.,6.25,4.4,2.4,
        r"$G_q(z)=\frac{T^2(z+1)}{2(z-1)^2}$"+"\n"+r"$G_v(z)=\frac{T}{z-1}$",22)
    wire(ax,[(1.25,y),(1.7,y)])
    wire(ax,[(5.,y),(6.02,y)])
    ax.text(5.76,y+.2,"+",fontsize=16)
    wire(ax,[(6.48,y),(8.,y)])
    ax.text(7.15,y+.26,r"$A(z)$",fontsize=16)
    wire(ax,[(12.4,y),(15.5,y)])
    ax.text(15.65,y,r"$Y(z)$",fontsize=18,va="center")
    vy = 6.8
    wire(ax,[(12.4,vy),(15.5,vy)])
    ax.text(15.65,vy,r"$V(z)$",fontsize=18,va="center")
    dot(ax,13.7,y)
    dot(ax,15.,vy)
    block(ax,8.35,4.75,4.05,.9,r"$(1+\alpha)K_p$",20)
    wire(ax,[(13.7,y),(13.7,5.2),(12.4,5.2)])
    wire(ax,[(8.35,5.2),(6.25,5.2),(6.25,7.87)])
    ax.text(6.54,7.62,"−",fontsize=19)
    block(ax,8.35,3.2,4.05,.9,r"$K_d$",20)
    wire(ax,[(15.,vy),(15.,3.65),(12.4,3.65)])
    wire(ax,[(8.35,3.65),(5.45,3.65),(5.45,7.55),(6.06,7.96)])
    ax.text(5.53,7.38,"−",fontsize=19)
    # No junction at the crossed output/position-feedback wires.
    ax.plot([13.7,13.7],[vy-.10,vy+.10],color="white",linewidth=4.)
    ax.plot([13.56,13.84],[vy,vy],color="black",linewidth=1.4)
    ax.text(.5,2.37,r"$\frac{Y(z)}{R(z)}=\frac{G_q(z)F_r(z)}"
        r"{1+(1+\alpha)K_pG_q(z)+K_dG_v(z)}$",fontsize=25)
    ax.text(.5,1.52,r"$D_{\rm cl}(z)=(z-1)^2+K_dT(z-1)"
        r"+\frac{1}{2}(1+\alpha)K_pT^2(z+1)$",fontsize=21)
    labels=[]
    for g, label in zip(gains,("Contact QP / 571","Whole-body QP / 577")):
        labels.append(f"{label}: Kp = {g['kp']:.3f} s^-2, Kd = {g['kd']:.3f} s^-1, alpha = {g['alpha']:.6f}")
    ax.text(.5,.73,"\n".join(labels),fontsize=11)
    ax.text(.5,.08,"Dcl is the homogeneous feedback characteristic; Fr adds reference memory z^-1. Reference velocity / acceleration clipping and QP changes are inactive.",fontsize=10)
    return fig


def nonlinear_page():
    fig, ax = page("V16  |  Actual nonlinear sampled control law",
        "Mathematical blocks for the trained whole-body QP controller; Q has 19 coordinates, v has 18 components, tau has 12.")
    block(ax,.5,6.6,5.2,2.2,
        r"$a_j=K_p(q_r-q)+K_d(v_r-\dot q)$"+"\n"+
        r"$+\alpha K_p(q_0-q)$"+"\n"+
        r"$a_{xy}=\ddot c_r+K_c(c_r-c)+D_c(\dot c_r-\dot c)$",16)
    ax.text(.55,9.02,r"$q_r,v_r,c_r,\dot c_r,\ddot c_r$",fontsize=17)
    wire(ax,[(2.8,9.2),(2.8,8.8)])
    block(ax,6.7,6.6,4.5,2.2,
        r"$(a^*,\hat\lambda^*)\simeq\Pi_{\mathcal{C}_k}(a_d)$"+"\n"+
        r"$\min\ \frac{1}{2}\|a-a_d\|_W^2$"+"\n"+
        r"$\quad+\frac{\epsilon_\tau}{2}\|\tau/L\|^2"
        r"+\frac{\epsilon_\lambda}{2}\|\lambda/(mg)\|^2$",17)
    wire(ax,[(5.7,7.7),(6.7,7.7)])
    ax.text(6.15,8.,r"$a_d[k]$",fontsize=14,ha="center")
    block(ax,12.2,6.6,4.2,2.2,
        r"$\tau_k^*=S(Ma^*+h-J_c^T\hat\lambda^*)$"+"\n"+
        r"$S=[\,0_{12\times6}\ \ I_{12}\,]$",16)
    wire(ax,[(11.2,7.7),(12.2,7.7)])
    block(ax,12.8,5.85,3.,.55,r"$\Gamma_T(\tau_k^*,\tau_{\rm prev})$",16)
    wire(ax,[(14.3,6.6),(14.3,6.4)])
    block(ax,12.8,4.5,3.,.85,r"$H_T(s)=\frac{1-e^{-sT}}{s}$",19)
    wire(ax,[(14.3,5.85),(14.3,5.35)])
    block(ax,12.8,3.1,3.,.85,r"$\Gamma_{5\rm ms}(\tau,\dot q)$",18)
    wire(ax,[(14.3,4.5),(14.3,3.95)])
    block(ax,.65,3.1,10.8,2.05,
        r"$\mathcal{P}:\quad M(Q)\dot v+h(Q,v)"
        r"=S^T\tau+J_{\rm phys}^T\lambda_{\rm phys}+d$"+"\n"+
        r"$\dot Q=N(Q)v$",20)
    wire(ax,[(12.8,3.525),(11.45,3.525)])
    ax.text(12.12,3.83,r"$\tau(t)$",fontsize=15,ha="center")
    wire(ax,[(.65,4.2),(.2,4.2),(.2,2.5),(15.8,2.5)])
    ax.text(16.,2.5,r"$Q,v$",fontsize=18,va="center")
    dot(ax,.2,4.2)
    wire(ax,[(.2,4.2),(.2,5.7),(1.1,5.7)])
    block(ax,1.1,5.275,1.4,.85,r"$t=kT$",15)
    wire(ax,[(2.5,5.7),(16.25,5.7)],arrow=False)
    ax.text(16.4,5.7,r"$x_k$",fontsize=16,va="center")
    for x in (2.8,9.,16.2):
        dot(ax,x,5.7)
        wire(ax,[(x,5.7),(x,6.6)])
    # Torque output crosses the sampled-state bus without a junction.
    ax.plot([14.3,14.3],[5.59,5.81],color="white",linewidth=4.)
    ax.plot([14.16,14.44],[5.7,5.7],color="black",linewidth=1.4)
    ax.text(.65,1.45,r"$\mathcal{C}_k:\ M_ba+h_b=J_{c,b}^T\lambda,\quad "
        r"J_ca=a_c,\quad\tau_{\min}\leq\tau\leq\tau_{\max}$",fontsize=19)
    ax.text(.65,.81,r"$0\leq\lambda_z\leq500,\quad "
        r"|\lambda_x|,|\lambda_y|\leq\frac{0.6}{\sqrt{2}}\lambda_z,"
        r"\quad\lambda_{\rm inactive}=0$",fontsize=19)
    ax.text(.65,.17,"IK/reference feedback and acceleration clipping are inside the sampled control law. QP uses 320 finite ADMM iterations.\n"
        "Gamma_T: 20 ms slew clipping before hold. Gamma_5ms: speed/effort envelope after hold. Model and physical contact forces are distinct.",fontsize=10)
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gains=[load_gains(job) for job in (571,577)]
    evidence=verify(gains)
    model_evidence=verify_saved_model(gains)
    # Preserve both earlier diagram sets before replacing the main PDF.
    primary=OUT/"v16_modeling_diagrams.pdf"
    archived=OUT/"v16_modeling_diagrams_architecture.pdf"
    if primary.is_file() and not archived.exists():
        shutil.copy2(primary,archived)
    appendix=OUT/"v16_control_transfer_diagrams.pdf"
    main_pages=[("v16_actual_wbc_blocks",actual_wbc_blocks_page()),
                ("v16_physical_force_balance",physical_force_page()),
                ("v16_physical_state_feedback",physical_state_page())]
    appendix_pages=[("v16_control_s_domain",continuous_page()),
                    ("v16_control_z_domain",discrete_page(gains)),
                    ("v16_control_nonlinear",nonlinear_page())]
    for target,pages in ((primary,main_pages),(appendix,appendix_pages)):
        temporary=target.with_suffix(".tmp.pdf")
        with PdfPages(temporary) as pdf:
            for name,fig in pages:
                fig.savefig(OUT/f"{name}.png",dpi=200,bbox_inches="tight")
                fig.savefig(OUT/f"{name}.svg",bbox_inches="tight")
                fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
                pdf.savefig(fig,bbox_inches="tight")
                plt.close(fig)
        temporary.replace(target)
    shutil.copy2(primary,OUT/"v16_control_block_diagrams.pdf")
    (OUT/"v16_control_math_verification.json").write_text(json.dumps({
        "source":"frozen controller helpers and final training checkpoints",
        "simulation_run":False,"renderer_sha256":digest(Path(__file__)),
        "verified_gain_records":evidence,
        "saved_source_model_identities":model_evidence,
        "main_diagram_scope":"actual WBC/controller/plant feedback first; floating-base force map second; fixed-base ideal equivalent third",
        "s_domain_scope":"ideal compensated joint axis; continuous-time equivalent",
        "z_domain_scope":"ideal acceleration-ZOH channel; measured velocity feedback; clipping inactive",
        "nonlinear_scope":"actual sampled WBC operator; finite numerical optimization; no global rational transfer function"
    },indent=2)+"\n")
    print(json.dumps({"pdf":str(primary),"appendix_pdf":str(appendix),"gain_records":evidence},indent=2))


if __name__=="__main__":
    main()
