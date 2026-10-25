"""Detect round holes (axis along a world axis) in a part mesh and fit bolt circles."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _fit(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    centre = sol[:2]
    r = np.sqrt(max(sol[2] + centre @ centre, 0.0))
    return centre, r, np.abs(np.linalg.norm(P - centre, axis=1) - r).max()


def holes(V, F, axis, r_range=(1.4, 3.2), max_span=25.0):
    """Short cylindrical wall patches perpendicular to ``axis``: [(centre(2), radius, (a0, a1))].

    Patches are grouped by centroid proximity; a group that does not fit one circle (e.g. a
    hole wall 0.8 mm from a larger bore) is regrouped at a smaller distance. Walls longer than
    ``max_span`` along the axis (e.g. a profile's axial bore) are not bolt holes and are skipped.
    """
    import scipy.sparse as sp
    a = np.eye(3)[axis]
    other = [i for i in range(3) if i != axis]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
    c = V[F].mean(axis=1)
    pending = [(np.flatnonzero(np.abs(n @ a) < 0.05), 1.2)]
    out = []
    while pending:
        idx, link = pending.pop()
        if len(idx) < 6:
            continue
        pairs = cKDTree(c[idx][:, other]).query_pairs(link, output_type='ndarray')
        g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
        ncomp, lab = connected_components(g, directed=False)
        for k in range(ncomp):
            sel = idx[lab == k]
            if len(sel) < 6:
                continue
            P = V[F[sel]].reshape(-1, 3)[:, other]
            centre, r, resid = _fit(P)
            span = V[F[sel]].reshape(-1, 3)[:, axis]
            if resid < 0.15:
                if r_range[0] <= r <= r_range[1] and span.max() - span.min() <= max_span:
                    out.append((centre, r, (span.min(), span.max())))
            elif link > 0.5 and ncomp > 0 and len(sel) < len(idx) or (link > 0.5 and ncomp == 1):
                pending.append((sel, round(link - 0.25, 2)))
    merged = []
    for h in out:
        if not any(np.linalg.norm(h[0] - m[0]) < 0.5 and abs(h[1] - m[1]) < 0.3 for m in merged):
            merged.append(h)
    return merged


def bolt_circles(hs, centre_guess=None):
    """Group holes by radius from a centre (given or mean) -> {(dia, pcd): count}."""
    if not hs:
        return {}
    C = np.array([h[0] for h in hs])
    centre = np.asarray(centre_guess) if centre_guess is not None else C.mean(axis=0)
    out = {}
    for c, r, span in hs:
        pcd = round(2 * np.linalg.norm(c - centre), 1)
        key = (round(2 * r, 2), pcd)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
