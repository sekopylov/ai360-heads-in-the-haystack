"""Line-by-line numpy twin of rh_core.gdn_implicit_attention, checked against the gated delta rule recurrence
(same update as transformers' torch_recurrent_gated_delta_rule). Runs without torch."""
import numpy as np
from scipy.linalg import solve_triangular


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def reference(q, k, v, beta, g):
    H, T, dk = q.shape
    S = np.zeros((H, dk, v.shape[-1]))
    out = np.zeros_like(v)
    for t in range(T):
        S = S * np.exp(g[:, t])[:, None, None]
        kv_mem = (S * k[:, t, :, None]).sum(-2)
        delta = (v[:, t] - kv_mem) * beta[:, t, None]
        S = S + k[:, t, :, None] * delta[:, None, :]
        out[:, t] = (S * q[:, t, :, None]).sum(-2)
    return out


def implicit(q, k, beta, g, q_start, chunk=64):
    H, T, d = q.shape
    n = T - q_start
    A = np.zeros((H, n, T))
    G = np.zeros((H, n, d))
    for i in range(T - 1, q_start - 1, -1):
        G[:, i - q_start] = q[:, i]
        ki = k[:, i]
        kG = np.einsum("hnd,hd->hn", G, ki)
        A[:, :, i] = beta[:, i, None] * kG
        G = np.exp(g[:, i])[:, None, None] * (G - (beta[:, i, None] * kG)[..., None] * ki[:, None, :])
    e = q_start
    while e > 0:
        s = max(0, e - chunk)
        C = e - s
        Kc, bc, gc = k[:, s:e], beta[:, s:e], g[:, s:e]
        incl = np.flip(np.cumsum(np.flip(gc, 1), 1), 1)
        excl = incl - gc
        M = np.triu(bc[..., None] * (Kc @ Kc.transpose(0, 2, 1)), 1) + np.eye(C)
        R = np.stack([solve_triangular(M[h], bc[h, :, None] * Kc[h], lower=False, unit_diagonal=True) for h in range(H)])
        RG = np.einsum("hcd,hnd->hnc", R, G)
        A[:, :, s:e] = RG * np.exp(excl)[:, None, :]
        G = np.exp(incl[:, :1])[..., None] * (G - np.einsum("hcd,hnc->hnd", Kc, RG))
        e = s
    return A


rng = np.random.default_rng(0)
H, T, dk, dv = 4, 400, 32, 24
q = l2norm(rng.standard_normal((H, T, dk))) * dk ** -0.5
k = l2norm(rng.standard_normal((H, T, dk)))
v = rng.standard_normal((H, T, dv))
beta = rng.random((H, T))
g = -rng.random((H, T)) * 0.1
ref = reference(q, k, v, beta, g)
for q_start, chunk in [(T - 1, 64), (T - 30, 64), (333, 16), (7, 5), (1, 64)]:
    A = implicit(q, k, beta, g, q_start, chunk)
    err = np.abs(np.einsum("hnt,htd->hnd", A, v) - ref[:, q_start:]).max()
    fut = max((np.abs(A[:, s, q_start + s + 1:]).max() if q_start + s + 1 < T else 0.0) for s in range(T - q_start))
    print(f"q_start={q_start:3d} chunk={chunk:2d}  max|o_rec-o_ref|={err:.2e}  future-leak={fut:.1e}")
    assert err < 1e-10 and fut == 0

# float32 at long context with strong retention (decay ~1): stability of the chunked solve
T2 = 6000
q2 = (l2norm(rng.standard_normal((2, T2, 128))) * 128 ** -0.5).astype(np.float32)
k2 = l2norm(rng.standard_normal((2, T2, 128))).astype(np.float32)
v2 = rng.standard_normal((2, T2, 128)).astype(np.float32)
b2 = rng.random((2, T2)).astype(np.float32)
g2 = (-rng.random((2, T2)) * 1e-3).astype(np.float32)
A32 = implicit(q2, k2, b2, g2, T2 - 3, 64)
ref2 = reference(q2.astype(np.float64), k2.astype(np.float64), v2.astype(np.float64), b2.astype(np.float64), g2.astype(np.float64))
rel = np.linalg.norm(np.einsum("hnt,htd->hnd", A32, v2) - ref2[:, T2 - 3:]) / np.linalg.norm(ref2[:, T2 - 3:])
print(f"float32, T={T2}, slow decay: relative error {rel:.2e}")
assert rel < 1e-4
print("OK")
