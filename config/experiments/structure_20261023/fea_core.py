"""Linear-elastic P2 tetrahedral FEA of one printed part.

Units: mm, N, MPa. Material is assigned per element: printed perimeter skin
(solid PLA) within ``skin`` mm of the surface and a homogenised infill core.
Interfaces are found by proximity to neighbouring parts. The body-side
interface is clamped; the foot-side interface carries the foot load as a
force plus moment spread over the contact faces (RBE3-style linear traction).
"""
from __future__ import annotations

import numpy as np
import pyamg
import tetgen
from scipy.spatial import cKDTree
from skfem import (Basis, BilinearForm, ElementTetP0, ElementTetP2, ElementVector, FacetBasis,
                   LinearForm, MeshTet, asm)
from skfem.helpers import ddot, dot, sym_grad, trace


def tet_mesh(V, F, maxvol, minratio=1.5):
    nodes, elems, *_ = tetgen.TetGen(np.asarray(V, float), np.asarray(F, np.int32)).tetrahedralize(
        order=1, minratio=minratio, maxvolume=maxvol, fixedvolume=True, quiet=True)
    return MeshTet(np.ascontiguousarray(nodes.T), np.ascontiguousarray(elems.T))


def surface_samples(V, F, spacing=0.6):
    """Points on a triangle soup at about ``spacing`` mm, for proximity queries."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    pts = [a, b, c, (a + b + c) / 3]
    longest = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                               np.linalg.norm(a - c, axis=1)]), axis=0)
    for n in np.unique(np.clip(np.ceil(longest / spacing), 1, 200).astype(int)):
        sel = np.ceil(longest / spacing).clip(1, 200).astype(int) == n
        if n < 2:
            continue
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                pts.append((i * a[sel] + j * b[sel] + k * c[sel]) / n)
    return np.concatenate(pts)


class TriSurface:
    """Exact point-to-triangle distance against a triangle mesh (candidates from a centroid KD-tree)."""

    def __init__(self, V, F):
        self.a, self.b, self.c = (np.asarray(V, float)[np.asarray(F)[:, i]] for i in range(3))
        cen = (self.a + self.b + self.c) / 3
        self.radius = np.max(np.stack([np.linalg.norm(x - cen, axis=1) for x in (self.a, self.b, self.c)]), axis=0)
        self.tree = cKDTree(cen)
        self.rmax = float(self.radius.max())

    def distance(self, P, cutoff):
        P = np.atleast_2d(P)
        out = np.full(len(P), np.inf)
        cand = self.tree.query_ball_point(P, cutoff + self.rmax)
        for i, idx in enumerate(cand):
            if idx:
                out[i] = _point_triangle_distance(P[i], self.a[idx], self.b[idx], self.c[idx]).min()
        return out


def _point_triangle_distance(p, a, b, c):
    """Distances from point p to triangles (a, b, c) [n,3] (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide='ignore', invalid='ignore'):
        v = vb / denom
        w = vc / denom
        q = a + ab * v[:, None] + ac * w[:, None]
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    regions = [
        ((d1 <= 0) & (d2 <= 0), a),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d6 >= 0) & (d5 <= d6), c),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t_ab[:, None]),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t_ac[:, None]),
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + (c - b) * t_bc[:, None]),
    ]
    done = np.zeros(len(a), bool)
    for mask, point in regions:
        m = mask & ~done
        q[m] = point[m]
        done |= m
    return np.linalg.norm(p - q, axis=1)


def contact_facets(mesh, neighbour, tol):
    """Boundary facets whose centroid and all vertices lie within ``tol`` of the neighbour TriSurface."""
    bf = mesh.boundary_facets()
    verts = mesh.facets[:, bf]
    uniq, inv = np.unique(verts, return_inverse=True)
    dv = neighbour.distance(mesh.p[:, uniq].T, tol)[inv].reshape(verts.shape)
    dc = neighbour.distance(mesh.p[:, verts].mean(axis=1).T, tol)
    return bf[(dc < tol) & (dv.max(axis=0) < tol)]


def facet_area(mesh, facets):
    p = mesh.p[:, mesh.facets[:, facets]]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], axis=0), axis=0)


class Part:
    def __init__(self, mesh, E, nu):
        """``E``: Young's modulus per element (MPa)."""
        self.mesh = mesh
        E = np.broadcast_to(np.asarray(E, float), (mesh.t.shape[1],))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.basis = Basis(mesh, ElementVector(ElementTetP2()))
        b0 = self.basis.with_element(ElementTetP0())
        lam_f, mu_f = b0.interpolate(self.lam), b0.interpolate(self.mu)

        @BilinearForm
        def elasticity(u, v, w):
            eu, ev = sym_grad(u), sym_grad(v)
            return 2.0 * w.mu * ddot(eu, ev) + w.lam * trace(eu) * trace(ev)

        self.K = asm(elasticity, self.basis, lam=lam_f, mu=mu_f).tocsr()
        self.volume = float(np.sum(self.basis.dx))

    def load_vector(self, facets, force, moment_about_centroid):
        """Linear traction on ``facets`` with resultant ``force`` and moment (about the area centroid)."""
        fb = FacetBasis(self.mesh, self.basis.elem, facets=facets)
        x = fb.global_coordinates().value  # (3, nf, nq)
        dA = fb.dx
        area = dA.sum()
        c = (x * dA).sum(axis=(1, 2)) / area
        r = x - c[:, None, None]
        J = np.zeros((3, 3))
        rr = (r * r).sum(axis=0)
        for i in range(3):
            for j in range(3):
                J[i, j] = ((rr * (i == j) - r[i] * r[j]) * dA).sum()
        omega = np.linalg.solve(J, moment_about_centroid)
        f0 = np.asarray(force, float) / area

        @LinearForm
        def traction(v, w):
            rr_ = w.x - c[:, None, None]
            t = f0[:, None, None] + np.cross(omega[:, None, None], rr_, axis=0)
            return dot(t, v)

        return asm(traction, fb), c, area

    def solve(self, loads, fixed_facets, tol=1e-9):
        """Solve K u = f for each load vector in ``loads`` (one AMG setup, CG per load)."""
        import scipy.sparse.linalg as spla
        D = self.basis.get_dofs(facets=fixed_facets).all()
        free = np.setdiff1d(np.arange(self.basis.N), D)
        A = self.K[free][:, free].tocsr()
        x_dof = self.basis.doflocs
        comps = self.basis.split_indices()
        B = np.zeros((self.basis.N, 6))
        for k, idx in enumerate(comps):
            B[idx, k] = 1.0
        X, Y, Z = (x_dof[i] - x_dof[i].mean() for i in range(3))
        # rotations about x, y, z: (0,-z,y), (z,0,-x), (-y,x,0)
        B[comps[1], 3], B[comps[2], 3] = -Z[comps[1]], Y[comps[2]]
        B[comps[0], 4], B[comps[2], 4] = Z[comps[0]], -X[comps[2]]
        B[comps[0], 5], B[comps[1], 5] = -Y[comps[0]], X[comps[1]]
        M = pyamg.smoothed_aggregation_solver(A, B=B[free], max_coarse=1000).aspreconditioner(cycle='V')
        out, self.iterations = [], []
        for f in loads:
            count = [0]
            uf, info = spla.cg(A, f[free], rtol=tol, maxiter=3000, M=M, callback=lambda _: count.__setitem__(0, count[0] + 1))
            if info != 0:
                raise RuntimeError(f'CG did not converge (info={info})')
            u = np.zeros(self.basis.N)
            u[free] = uf
            out.append(u)
            self.iterations.append(count[0])
        return out

    def stresses(self, u):
        """Element-mean stress tensors (nel, 3, 3), von Mises and principal stresses."""
        G = self.basis.interpolate(u).grad  # (3, 3, nel, nq)
        eps = 0.5 * (G + np.swapaxes(G, 0, 1))
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        sig = 2 * self.mu[None, None, :, None] * eps
        for i in range(3):
            sig[i, i] += self.lam[:, None] * tr
        S = sig.mean(axis=3).transpose(2, 0, 1)
        dev = S - np.eye(3)[None] * (np.trace(S, axis1=1, axis2=2) / 3)[:, None, None]
        vm = np.sqrt(1.5 * np.einsum('nij,nij->n', dev, dev))
        principal = np.linalg.eigvalsh(S)
        return S, vm, principal

    def vertex_displacement(self, u):
        nd = self.basis.nodal_dofs  # (3, nverts)
        return u[nd]

    def rigid_fit(self, u, facets):
        """Best-fit translation a (at c) and rotation theta of the vertices of ``facets``."""
        verts = np.unique(self.mesh.facets[:, facets])
        x = self.mesh.p[:, verts].T
        d = self.vertex_displacement(u)[:, verts].T
        c = x.mean(axis=0)
        r = x - c
        A = np.zeros((3 * len(x), 6))
        A[0::3, 0] = A[1::3, 1] = A[2::3, 2] = 1
        # theta x r = (ty rz - tz ry, tz rx - tx rz, tx ry - ty rx)
        A[0::3, 4], A[0::3, 5] = r[:, 2], -r[:, 1]
        A[1::3, 3], A[1::3, 5] = -r[:, 2], r[:, 0]
        A[2::3, 3], A[2::3, 4] = r[:, 1], -r[:, 0]
        sol = np.linalg.lstsq(A, d.ravel(), rcond=None)[0]
        return sol[:3], sol[3:], c
