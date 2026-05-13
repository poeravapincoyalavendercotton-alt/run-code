"""Oracle for Finsler geodesic asymmetry task."""

import base64
import io
import sys
import numpy as np

_EMBEDDED = (
    "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"
    "AAAA"
)


def _decode():
    raw = base64.b64decode(_EMBEDDED)
    z = np.load(io.BytesIO(raw))
    return z["X_pca"].astype(np.float64), z["labels"].astype(np.int32)


def get_data():
    X_pca, labels = _decode()
    return {"X_pca": X_pca, "labels": labels}


def get_vk_score(sigma_v: float) -> float:
    if sigma_v <= 0:
        raise ValueError("sigma_v must be positive")
    global _vk_cache
    if "_vk_cache" not in globals() or _vk_cache is None:
        from sklearn.neighbors import NearestNeighbors
        X_pca, labels = _decode()
        n = len(X_pca)
        nbrs = NearestNeighbors(n_neighbors=15, metric="euclidean").fit(X_pca)
        _, knn_inds = nbrs.kneighbors(X_pca)
        labels_arr = labels.astype(np.int32)
        src_idx, dst_idx = [], []
        for i in range(n):
            li = labels_arr[i]
            if li == 2:
                continue
            for j in knn_inds[i]:
                if labels_arr[j] == li + 1:
                    src_idx.append(i); dst_idx.append(int(j))
        src_idx = np.asarray(src_idx, dtype=np.int64)
        dst_idx = np.asarray(dst_idx, dtype=np.int64)
        stage0_next = X_pca[labels_arr == 1]
        stage1_next = X_pca[labels_arr == 2]
        _vk_cache = {
            "X_pca": X_pca, "labels": labels_arr, "n": n,
            "src_idx": src_idx, "dst_idx": dst_idx,
            "stage_next_for": {0: stage0_next, 1: stage1_next},
        }
    c = _vk_cache
    X = c["X_pca"]; labs = c["labels"]
    sv2 = 2.0 * sigma_v * sigma_v
    vel = np.zeros_like(X)
    for li, nxt in c["stage_next_for"].items():
        mask = labs == li
        Xs = X[mask]
        diff = nxt[None, :, :] - Xs[:, None, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        w = np.exp(-d2 / sv2)
        s = w.sum(axis=1, keepdims=True)
        s = np.where(s > 1e-12, s, 1.0)
        vel[mask] = (w[:, :, None] * diff).sum(axis=1) / s
    src = c["src_idx"]; dst = c["dst_idx"]
    if len(src) == 0:
        return 0.0
    vi = vel[src]; dv = X[dst] - X[src]
    nvi = np.linalg.norm(vi, axis=1)
    nd = np.linalg.norm(dv, axis=1)
    valid = (nvi > 1e-10) & (nd > 1e-10)
    if not valid.any():
        return 0.0
    cos = np.einsum("ij,ij->i", vi[valid], dv[valid]) / (nvi[valid] * nd[valid])
    return float(cos.mean())


def submit_answer(answer: str) -> None:
    print(f"Submitted: {answer}")


_CALL_LIMITS = {"get_vk_score": 20}
_call_counts = {"get_vk_score": 0}
_vk_cache = None


def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def handle_query(mode, parameters=None):
    if parameters is None:
        parameters = {}
    if mode == "help":
        result = {"modes": {"get_data": {}, "get_vk_score": {}, "budget": {}}}
    elif mode == "get_data":
        d = get_data()
        result = {"X_pca": d["X_pca"].tolist(), "labels": d["labels"].tolist()}
    elif mode == "budget":
        result = {k: max(0, _CALL_LIMITS[k] - _call_counts[k]) for k in _CALL_LIMITS}
    elif mode == "get_vk_score":
        if _call_counts["get_vk_score"] >= _CALL_LIMITS["get_vk_score"]:
            raise RuntimeError("get_vk_score limit reached")
        sv = parameters.get("sigma_v")
        if sv is None:
            raise ValueError("missing sigma_v")
        _call_counts["get_vk_score"] += 1
        result = float(get_vk_score(float(sv)))
    else:
        raise ValueError(f"Unknown mode {mode}")
    return _make_serializable(result)
