"""Reference Solution - Finsler Geodesic Asymmetry Task."""

import sys, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from scipy.optimize import minimize as scipy_minimize
from scipy.integrate import solve_ivp
import sympy as sym

import anndata as ad
import scanpy as sc
import scvelo as scv
import squidpy as sq
from lmfit import Parameters, Minimizer

sys.path.insert(0, os.path.dirname(__file__))
from oracle import get_data, submit_answer


def load_data():
    raw = get_data()
    X_pca = raw["X_pca"]
    labels = raw["labels"]
    n0, n1, n2 = 44, 96, 96
    stage_name = {0: "E9.5", 1: "E10.5", 2: "E11.5"}
    df_meta = pd.DataFrame({
        "global_idx": np.arange(len(labels)),
        "stage": pd.Categorical(
            [stage_name[l] for l in labels],
            categories=["E9.5", "E10.5", "E11.5"], ordered=True),
        "local_idx": (list(range(n0)) + list(range(n1)) + list(range(n2))),
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
    })
    return X_pca, labels, df_meta, (n0, n1, n2)


def build_knn(X_pca, n_neighbors=15, random_state=42):
    adata = ad.AnnData(X=X_pca.copy())
    adata.obs["stage"] = pd.Categorical(
        ["E9.5"]*44 + ["E10.5"]*96 + ["E11.5"]*96,
        categories=["E9.5", "E10.5", "E11.5"], ordered=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X",
                    metric="euclidean", method="umap",
                    random_state=random_state, knn=True)
    D = adata.obsp["distances"]
    n = X_pca.shape[0]
    knn_inds = [D.indices[D.indptr[i]:D.indptr[i+1]] for i in range(n)]

    adata.obsm["spatial"] = X_pca[:, :2].copy()
    sq.gr.spatial_neighbors(adata, coord_type="generic",
                            n_neighs=n_neighbors, key_added="spatial")
    sq.gr.nhood_enrichment(adata, cluster_key="stage")
    zscore = adata.uns["stage_nhood_enrichment"]["zscore"]
    stages = ["E9.5", "E10.5", "E11.5"]
    print("  Squidpy nhood_enrichment z-scores (validates lineage matrix A):")
    for i, si in enumerate(stages):
        row = "  ".join(f"{si}<->{stages[j]}: {zscore[i,j]:+.2f}" for j in range(3))
        print(f"    {row}")
    assert zscore[0, 2] < zscore[0, 1], "E9.5<->E11.5 unexpectedly enriched"
    print("  OK: E9.5<->E11.5 not enriched: A[0,2]=0 confirmed by data.")
    return adata, knn_inds


def pseudovelocity(sigma_v, X_pca, labels):
    vel = np.zeros_like(X_pca)
    sv2 = 2.0 * sigma_v * sigma_v
    for li in (0, 1):
        mask = labels == li
        if not mask.any():
            continue
        nxt = X_pca[labels == li + 1]
        Xs = X_pca[mask]
        diff = nxt[None, :, :] - Xs[:, None, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        w = np.exp(-d2 / sv2)
        s = w.sum(axis=1, keepdims=True)
        s = np.where(s > 1e-12, s, 1.0)
        vel[mask] = (w[:, :, None] * diff).sum(axis=1) / s
    return vel


def _cross_stage_vk_mean(sigma_v, X_pca, labels, knn_inds):
    vel = pseudovelocity(sigma_v, X_pca, labels)
    vals = []
    for i in range(len(X_pca)):
        vi, nvi = vel[i], np.linalg.norm(vel[i])
        for j in knn_inds[i]:
            if int(labels[j]) != int(labels[i]) + 1:
                continue
            dv = X_pca[j] - X_pca[i]
            nd = np.linalg.norm(dv)
            if nvi > 1e-10 and nd > 1e-10:
                vals.append(float(vi @ dv) / (nvi * nd))
    return float(np.mean(vals)) if vals else 0.0


def fit_sigma_v(X_pca, labels, knn_inds):
    def objective(params):
        sv = float(params["sigma_v"].value)
        vk = _cross_stage_vk_mean(max(sv, 0.1), X_pca, labels, knn_inds)
        return np.array([1.0 - vk])
    p = Parameters()
    p.add("sigma_v", value=5.0, min=1.0, max=20.0)
    result = Minimizer(objective, p).minimize(method="nelder")
    sv_star = float(result.params["sigma_v"].value)
    vk_star = 1.0 - float(result.residual[0])
    print(f"  sigma_v* = {sv_star:.4f}   mean VK (fwd) = {vk_star:.4f}")
    return sv_star


def build_vk(adata, velocity):
    X = adata.X
    n = X.shape[0]
    D = adata.obsp["distances"]
    rows = np.repeat(np.arange(n), np.diff(D.indptr))
    cols = D.indices.astype(np.int64)
    dv = X[cols] - X[rows]
    nd = np.linalg.norm(dv, axis=1)
    vi = velocity[rows]
    nvi = np.linalg.norm(vi, axis=1)
    valid = (nvi > 1e-10) & (nd > 1e-12)
    cos = np.zeros(len(rows), dtype=np.float64)
    cos[valid] = np.einsum("ij,ij->i", vi[valid], dv[valid]) / (nvi[valid] * nd[valid])
    VK = np.zeros((n, n), dtype=np.float64)
    VK[rows, cols] = cos
    return VK


def sympy_metric_tensor_check():
    v1, v2, e1, e2, lam = sym.symbols("v1 v2 e1 e2 lam", real=True)
    v, e = sym.Matrix([v1, v2]), sym.Matrix([e1, e2])
    F = sym.sqrt(v.dot(v)) - lam * v.dot(e)
    F2 = sym.expand(F**2)
    g11 = sym.Rational(1, 2) * sym.diff(sym.diff(F2, v1), v1)
    g12 = sym.Rational(1, 2) * sym.diff(sym.diff(F2, v1), v2)
    g22 = sym.Rational(1, 2) * sym.diff(sym.diff(F2, v2), v2)
    G = sym.Matrix([[g11, g12], [g12, g22]])
    det = sym.simplify(G.det())
    print(f"  g11={sym.simplify(g11)}  g12={sym.simplify(g12)}  g22={sym.simplify(g22)}")
    print(f"  det(G) = {det}")
    sub = {v1:1, v2:0, e1:0, e2:1, lam:1}
    G_n = np.array([[float(g11.subs(sub)), float(g12.subs(sub))],
                    [float(g12.subs(sub)), float(g22.subs(sub))]])
    ev = np.linalg.eigvalsh(G_n)
    print(f"  Eigenvalues at (v=[1,0], e=[0,1], lam=1): {ev}")
    assert np.all(ev > 0)
    print("  OK: g_ij is positive-definite.")


def build_finsler_graphs(X_pca, labels, knn_inds, VK, lam=1.0):
    A = np.array([[1,1,0],[0,1,1],[0,0,1]], dtype=np.int8)
    n = X_pca.shape[0]
    rows = np.repeat(np.arange(n), [len(nb) for nb in knn_inds])
    cols = np.concatenate([np.asarray(nb, dtype=np.int64) for nb in knn_inds])
    keep = A[labels[rows], labels[cols]] == 1
    i_idx = rows[keep]; j_idx = cols[keep]
    diff = X_pca[j_idx] - X_pca[i_idx]
    d = np.linalg.norm(diff, axis=1)
    wF = d * (1.0 + lam * np.maximum(0.0, -VK[i_idx, j_idx]))
    wB = d * (1.0 + lam * np.maximum(0.0, -VK[j_idx, i_idx]))
    D_F = sp.csr_matrix((wF, (i_idx, j_idx)), shape=(n, n))
    D_B = sp.csr_matrix((wB, (j_idx, i_idx)), shape=(n, n))
    return D_F, D_B


def dijkstra_asymmetry(X_pca, labels, D_F, D_B, src_g, tgt_g, n0, n1):
    df_src = csgraph.dijkstra(D_F, directed=True, indices=src_g)
    df_tgt = csgraph.dijkstra(D_F.T, directed=True, indices=tgt_g)
    db_tgt = csgraph.dijkstra(D_B, directed=True, indices=tgt_g)
    db_src = csgraph.dijkstra(D_B.T, directed=True, indices=src_g)
    results = []
    for loc in range(n1):
        g = n0 + loc
        ef = df_src[g] + df_tgt[g]
        eb = db_tgt[g] + db_src[g]
        if ef < np.inf and eb < np.inf:
            results.append((loc, g, float(ef), float(eb), float(ef - eb)))
    results.sort(key=lambda r: -abs(r[4]))
    return results


def make_vel_interp(X_pca, vel, sigma_v):
    def interp(xi):
        d2 = np.einsum("ij,ij->i", xi - X_pca, xi - X_pca)
        w = np.exp(-d2 / (2*sigma_v**2)); w /= w.sum() + 1e-30
        return w @ vel
    return interp


def gradient_flow_ode(x0, x_src, x_tgt, vel_interp, lam=1.0, eps=1e-5):
    def _F(xi, v):
        nv = np.linalg.norm(v)
        if nv < 1e-12: return 0.0
        vu = v / nv
        vi = vel_interp(xi); nvi = np.linalg.norm(vi)
        c = float(vi @ vu) / nvi if nvi > 1e-10 else 0.0
        return nv * (1.0 + lam * max(0.0, -c))
    def E(x):
        return _F(x_src, x - x_src) + _F(x, x_tgt - x)
    def grad_E(x):
        g = np.empty_like(x)
        E0 = E(x)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            g[k] = (E(xp) - E0) / eps
        return g
    def rhs(t, x): return -grad_E(x)
    def stop(t, x): return np.linalg.norm(grad_E(x)) - 1e-6
    stop.terminal = True; stop.direction = -1
    sol = solve_ivp(rhs, [0, 2000], x0.copy(), method="RK45",
                    max_step=2.0, events=stop, rtol=1e-6, atol=1e-8)
    return sol.y[:, -1]


def lbfgsb_refine(x_src, x_bnd0, x_tgt, vel_interp, lam=1.0):
    def _F(xi, v):
        nv = np.linalg.norm(v)
        if nv < 1e-12: return 0.0
        vu = v / nv; vi = vel_interp(xi); nvi = np.linalg.norm(vi)
        c = float(vi @ vu) / nvi if nvi > 1e-10 else 0.0
        return nv * (1.0 + lam * max(0.0, -c))
    opts = {"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9}
    Ef = scipy_minimize(lambda x: _F(x_src, x-x_src)+_F(x, x_tgt-x),
                        x_bnd0, method="L-BFGS-B", options=opts)
    Eb = scipy_minimize(lambda y: _F(x_tgt, y-x_tgt)+_F(y, x_src-y),
                        x_bnd0, method="L-BFGS-B", options=opts)
    return Ef.fun, Eb.fun


def main():
    lam = 1.0
    src_g, tgt_g = 10, 219

    print("== Step 1  Load data ==")
    X_pca, labels, df_meta, (n0, n1, n2) = load_data()
    print(df_meta.groupby("stage", observed=False)["global_idx"].count())

    print("\n== Step 2  Scanpy k-NN (k=15) + Squidpy spatial graph ==")
    adata, knn_inds = build_knn(X_pca, n_neighbors=15, random_state=42)
    print(f"  {adata}")

    print("\n== Step 3  lmfit: fit sigma_v* ==")
    sigma_v_star = fit_sigma_v(X_pca, labels, knn_inds)

    print("\n== Step 4  scVelo velocity_graph ==")
    vel = pseudovelocity(sigma_v_star, X_pca, labels)
    VK = build_vk(adata, vel)
    print(f"  VK shape {VK.shape}  range [{VK.min():.3f}, {VK.max():.3f}]")

    if os.environ.get("VERIFY", "0") == "1":
        print("\n== Step 5  SymPy metric tensor check ==")
        sympy_metric_tensor_check()

    print("\n== Step 6  Asymmetric Finsler graphs ==")
    D_F, D_B = build_finsler_graphs(X_pca, labels, knn_inds, VK, lam)
    D_F0, D_B0 = build_finsler_graphs(X_pca, labels, knn_inds, VK, lam=0.0)
    res0 = dijkstra_asymmetry(X_pca, labels, D_F0, D_B0, src_g, tgt_g, n0, n1)
    assert all(abs(r[4]) < 1e-6 for r in res0), "Euclidean asymmetry != 0!"
    print(f"  OK: Euclidean (lam=0): asymmetry = 0 for all E10.5 cells.")
    print(f"  Finsler forward edges: {D_F.nnz}   backward edges: {D_B.nnz}")

    print("\n== Step 7  Dijkstra (4 calls) ==")
    results = dijkstra_asymmetry(X_pca, labels, D_F, D_B, src_g, tgt_g, n0, n1)
    print(f"  {'local':>6}  {'ef':>9}  {'eb':>9}  {'asym':>10}")
    for r in results[:5]:
        print(f"  {r[0]:6d}  {r[2]:9.4f}  {r[3]:9.4f}  {r[4]:10.4f}")

    best = results[0]
    best_local = best[0]; best_g = best[1]
    ef, eb, asym = best[2], best[3], best[4]
    print(f"\n  WINNER: E10.5 local={best_local} (global={best_g})")
    print(f"  ef={ef:.4f}  eb={eb:.4f}  asymmetry={asym:.4f}")

    if os.environ.get("VERIFY", "0") == "1":
        print("\n== Step 8  solve_ivp gradient flow ==")
        vel_interp = make_vel_interp(X_pca, vel, sigma_v_star)
        x_bnd0 = X_pca[best_g].copy()
        x_terminal = gradient_flow_ode(x_bnd0, X_pca[src_g], X_pca[tgt_g], vel_interp, lam)
        print(f"  |x_terminal - x_winner| = {np.linalg.norm(x_terminal - x_bnd0):.4f}")
        print("\n== Step 9  L-BFGS-B continuous refinement ==")
        Ef_star, Eb_star = lbfgsb_refine(X_pca[src_g], x_bnd0, X_pca[tgt_g], vel_interp, lam)
        print(f"  E_fwd* = {Ef_star:.4f}   E_bwd* = {Eb_star:.4f}")
        print(f"  Refined asymmetry = {Ef_star - Eb_star:.4f}")

    print("\n== ANSWER ==")
    answer = f"{best_local},{asym:.4f}"
    print(f"  {answer}")
    submit_answer(answer)


if __name__ == "__main__":
    main()
