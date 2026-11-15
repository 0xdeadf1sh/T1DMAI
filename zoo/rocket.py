"""MiniRocket (Dempster et al. 2021): random dilated kernels over 6 h, then ridge per zone step.

Channels are glucose, insulin and carbohydrate; bands are residual quantiles on the cal block.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from config import QUANTILE_LEVELS
from zoo.classical import Forecaster, ridge_features
from zoo.windows import ZONE, Windows

KERNEL_LEN = 9
INDICES = np.array(list(combinations(range(KERNEL_LEN), 3)), np.int64)  # the 84 +2 taps
N_KERNELS = len(INDICES)
ALPHA_GRID = np.logspace(-4, 3, 15)  # times the fit row count: unit-variance columns
EXTRA_ALPHA = 1.0  # the hand features' penalty, as ridge ARX's
GRID_STEP = 1024  # inputs on a 1/1024 grid within +-64: every response sum is exact in float32
GRID_MAX = 64.0


def quantize(x: np.ndarray) -> np.ndarray:
    """Standardized channels -> exact grid values, so features do not depend on summation order."""
    return np.clip(np.round(x * GRID_STEP) / GRID_STEP, -GRID_MAX, GRID_MAX).astype(np.float32)


def fit_dilations(n_timepoints: int, n_features: int, max_dilations: int = 32):
    """``(dilations, features per dilation)``, both int64; the paper's exponential spacing."""
    per_kernel = n_features // N_KERNELS
    n_dil = min(per_kernel, max_dilations)
    exp = np.log2((n_timepoints - 1) / (KERNEL_LEN - 1))
    dil, count = np.unique(np.logspace(0, exp, n_dil, base=2).astype(np.int32), return_counts=True)
    count = (count * (per_kernel / n_dil)).astype(np.int32)
    for i in range(per_kernel - int(count.sum())):
        count[i % len(count)] += 1
    return dil.astype(np.int64), count.astype(np.int64)


def golden_quantiles(n: int) -> np.ndarray:
    return np.array([(i * (np.sqrt(5) + 1) / 2) % 1 for i in range(1, n + 1)], np.float32)


def _responses(x: torch.Tensor, d: int) -> torch.Tensor:
    """(B, C, T) -> (B, C, 84, T): weights -1 with +2 at three taps, dilation ``d``, zero pad."""
    T = x.shape[-1]
    pad = (KERNEL_LEN - 1) * d // 2
    xp = torch.nn.functional.pad(x, (pad, pad))
    taps = torch.stack([xp[..., i * d:i * d + T] for i in range(KERNEL_LEN)], dim=2)
    idx = torch.as_tensor(INDICES, device=x.device)
    return 3 * taps[:, :, idx].sum(3) - taps.sum(2, keepdim=True)


def transform(x: torch.Tensor, p: dict) -> torch.Tensor:
    """(B, C, T) -> (B, features) float64 proportions of positive values, aeon's feature order."""
    T = x.shape[-1]
    out, comb, f0 = [], 0, 0
    for j, (d, nf) in enumerate(zip(p['dilations'].tolist(), p['per_dilation'].tolist())):
        pad = (KERNEL_LEN - 1) * d // 2
        mask = torch.as_tensor(p['channels'][comb:comb + N_KERNELS], dtype=x.dtype, device=x.device)
        c = torch.einsum('bckt,kc->bkt', _responses(x, d), mask)
        b = torch.as_tensor(p['biases'][f0:f0 + N_KERNELS * nf], dtype=x.dtype, device=x.device)
        b = b.reshape(N_KERNELS, 1, nf)
        full = (c[..., None] > b).sum(2, dtype=torch.int16)
        inner = (c[:, :, pad:T - pad, None] > b).sum(2, dtype=torch.int16)
        odd = (j + np.arange(N_KERNELS)) % 2 == 1
        recip = torch.as_tensor(np.where(odd, 1.0 / (T - 2 * pad), 1.0 / T)[:, None], device=x.device)
        count = torch.where(torch.as_tensor(odd, device=x.device)[None, :, None], inner, full)
        out.append((count.double() * recip).reshape(len(x), -1))
        comb += N_KERNELS
        f0 += N_KERNELS * nf
    return torch.cat(out, dim=1)


def fit_kernels(x: np.ndarray, n_features: int, rng: np.random.Generator) -> dict:
    """Channel subsets per (dilation, kernel), and biases at golden-ratio quantiles of one
    random window's response each. ``x`` is (N, C, T) float32."""
    n, C, T = x.shape
    dil, per = fit_dilations(T, n_features)
    n_comb = N_KERNELS * len(dil)
    k = (2 ** rng.uniform(0, np.log2(min(C, 9) + 1), n_comb)).astype(np.int64)
    channels = np.zeros((n_comb, C), bool)
    for i in range(n_comb):
        channels[i, rng.choice(C, k[i], replace=False)] = True
    q = golden_quantiles(N_KERNELS * int(per.sum()))
    biases = np.empty(len(q), np.float32)
    pick = rng.integers(0, n, n_comb)
    f0 = 0
    for j, (d, nf) in enumerate(zip(dil.tolist(), per.tolist())):
        rows = slice(j * N_KERNELS, (j + 1) * N_KERNELS)
        r = _responses(torch.from_numpy(x[pick[rows]]).double(), d).numpy()
        for kk in range(N_KERNELS):
            c = (r[kk, :, kk] * channels[rows][kk][:, None]).sum(0)
            biases[f0:f0 + nf] = np.quantile(c, q[f0:f0 + nf])
            f0 += nf
    return {'dilations': dil, 'per_dilation': per, 'channels': channels, 'biases': biases}


def z_gram(A: torch.Tensor, mu: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """Raw ``[1, f]`` Gram -> Gram of ``[1, (f - mu) * inv]``, O(p^2)."""
    n, s, B = A[0, 0], A[0, 1:], A[1:, 1:]
    G = torch.empty_like(A)
    G[0, 0] = n
    G[0, 1:] = G[1:, 0] = (s - n * mu) * inv
    G[1:, 1:] = (B - torch.outer(mu, s) - torch.outer(s, mu) + n * torch.outer(mu, mu)) \
        * torch.outer(inv, inv)
    return G


def z_rhs(R: torch.Tensor, mu: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """Raw ``[1, f]^T y`` -> ``[1, z]^T y``."""
    return torch.cat([R[:1], (R[1:] - mu[:, None] * R[:1]) * inv[:, None]])


def standardizer(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and 1/sd from a raw Gram; constant columns get 1/sd = 0."""
    n = A[0, 0]
    mu = A[0, 1:] / n
    var = torch.diagonal(A)[1:] / n - mu * mu
    return mu, torch.where(var > 1e-10, var.clamp_min(1e-10).rsqrt(), torch.zeros_like(var))


def penalty(n_cols: int, n_rocket: int, alpha: float, like: torch.Tensor) -> torch.Tensor:
    """Diagonal penalty on ``[1, rocket, extras]``: 0, ``alpha``, ``EXTRA_ALPHA``."""
    pen = torch.full((n_cols,), EXTRA_ALPHA, dtype=like.dtype, device=like.device)
    pen[0] = 0.0
    pen[1:1 + n_rocket] = alpha
    return pen


def ridge_path(A: torch.Tensor, R: torch.Tensor, mu, inv, n_rocket: int, alphas):
    """Yield ``[1, z]`` coefficients (1 + p, targets) per rocket ``alpha``; one eigendecomposition
    of the rocket block after the intercept and extras are solved out exactly."""
    G = z_gram(A, mu, inv)
    Rz = z_rhs(R, mu, inv)
    r = slice(1, 1 + n_rocket)
    e = torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(1 + n_rocket, len(G))]).to(G.device)
    Gee = G[e][:, e] + torch.diag(penalty(len(G), n_rocket, 0.0, G)[e])
    Ger = G[e][:, r]
    L = torch.linalg.cholesky(Gee)
    X = torch.cholesky_solve(Ger, L)
    xb = torch.cholesky_solve(Rz[e], L)
    S = G[r, r] - Ger.T @ X
    bs = Rz[r] - Ger.T @ xb
    del G
    s, V = torch.linalg.eigh(S)
    del S
    Vb = V.T @ bs
    for a in alphas:
        coef = torch.empty_like(Rz)
        coef[r] = V @ (Vb / (s[:, None] + a))
        coef[e] = xb - X @ coef[r]
        yield coef


def solve_step(A: torch.Tensor, r: torch.Tensor, mu, inv, n_rocket: int,
               alpha: float) -> torch.Tensor:
    """Ridge on ``[1, z]`` under ``penalty``; ``A``, ``r`` hold only the rows used."""
    G = z_gram(A, mu, inv)
    G.diagonal().add_(penalty(len(G), n_rocket, alpha, G))
    rz = z_rhs(r[:, None], mu, inv)
    return torch.cholesky_solve(rz, torch.linalg.cholesky(G))[:, 0]


class MiniRocket(Forecaster):
    name = 'minirocket'

    def __init__(self, n_features: int = 10_000, ctx: int = 72, seed: int = 0,
                 chunk: int = 512) -> None:
        self.n_features, self.ctx, self.seed, self.chunk = n_features, ctx, seed, chunk
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.kernels: dict = {}
        self.ch_mu = self.ch_sd = self.offsets = self.alphas = np.empty(0)
        self.mu = self.inv = self.coef = torch.empty(0)

    @property
    def n_rocket(self) -> int:
        return N_KERNELS * int(self.kernels['per_dilation'].sum())

    def _channels(self, w: Windows) -> np.ndarray:
        """(N, 3, ctx) standardized glucose, insulin and carbohydrate on the exact grid."""
        x = np.stack([w.bg, w.insulin[:, :self.ctx], w.carb[:, :self.ctx]], axis=1)
        return quantize((x - self.ch_mu[None, :, None]) / self.ch_sd[None, :, None])

    def _raw(self, w: Windows, idx: np.ndarray) -> torch.Tensor:
        """(len(idx), 1 + features) float64 rows ``[1, rocket, last bg, dose bins]``."""
        part = w.take(idx)
        r = transform(torch.from_numpy(self._channels(part)).to(self.device), self.kernels).double()
        t = [torch.from_numpy(np.asarray(a, np.float64)).to(self.device)
             for a in (part.bg, part.insulin, part.carb)]
        extra = ridge_features(*t, lags=1, exogenous=True)
        return torch.cat([torch.ones(len(idx), 1, dtype=r.dtype, device=self.device), r, extra], 1)

    def _chunks(self, idx: np.ndarray):
        return (idx[i:i + self.chunk] for i in range(0, len(idx), self.chunk))

    def _gram(self, w: Windows, idx: np.ndarray, y0: np.ndarray | None):
        A = R = None
        for part in self._chunks(idx):
            F = self._raw(w, part)
            A = F.T @ F if A is None else A.addmm_(F.T, F)
            if y0 is not None:
                y = torch.from_numpy(y0[part]).to(self.device)
                R = F.T @ y if R is None else R.addmm_(F.T, y)
        return A, R

    def _delta(self, w: Windows, coef: torch.Tensor) -> np.ndarray:
        """(N, k) change from the last reading; ``coef`` is (1 + p, k) on ``[1, z]``."""
        out = []
        for part in self._chunks(np.arange(len(w))):
            F = self._raw(w, part)
            Z = torch.cat([F[:, :1], (F[:, 1:] - self.mu) * self.inv], 1)
            out.append((Z @ coef).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, coef.shape[1]))

    def fit(self, w: Windows, val: Windows | None = None) -> None:
        if val is None or len(val) < 2:
            raise ValueError('minirocket picks its ridge penalty on validation windows')
        rng = np.random.default_rng(self.seed)
        chans = (w.bg, w.insulin[:, :self.ctx], w.carb[:, :self.ctx])
        self.ch_mu = np.array([c.mean(dtype=np.float64) for c in chans], np.float32)
        self.ch_sd = np.array([c.std(dtype=np.float64) + 1e-6 for c in chans], np.float32)
        self.kernels = fit_kernels(self._channels(w), self.n_features, rng)
        y = w.target.astype(np.float64) - w.bg[:, -1:]
        seen = np.isfinite(y)
        y0 = np.where(seen, y, 0.0)
        A_all, R = self._gram(w, np.arange(len(w)), y0)
        inc = np.flatnonzero(~seen.all(1))
        A_c, R_c = A_all.clone(), R.clone()
        if len(inc):
            A_inc, R_inc = self._gram(w, inc, y0)
            A_c -= A_inc
            R_c -= R_inc
            del A_inc
        self.mu, self.inv = standardizer(A_c)
        self.alphas = self._select_alpha(A_c, R_c, val)
        del A_c
        coef = torch.empty(A_all.shape[0], ZONE, dtype=A_all.dtype, device=self.device)
        for k in range(ZONE):
            miss = np.flatnonzero(~seen[:, k])
            A_k = A_all - self._gram(w, miss, None)[0] if len(miss) else A_all.clone()
            coef[:, k] = solve_step(A_k, R[:, k], self.mu, self.inv, self.n_rocket,
                                    float(self.alphas[k]))
            del A_k
        self.coef = coef

    def _select_alpha(self, A_c, R_c, val: Windows) -> np.ndarray:
        """Per zone step, the grid alpha with the lowest RMSE on ``val``."""
        grid = float(A_c[0, 0]) * ALPHA_GRID
        path = torch.stack(list(ridge_path(A_c, R_c, self.mu, self.inv, self.n_rocket, grid)), 1)
        pred = self._delta(val, path.reshape(len(path), -1)).reshape(len(val), len(grid), ZONE)
        truth = (val.target.astype(np.float64) - val.bg[:, -1:])[:, None, :]
        se = (pred - truth) ** 2
        n = np.isfinite(se).sum(0)
        if (n == 0).any():
            raise ValueError('a zone step has no measured validation target')
        err = np.sqrt(np.nansum(se, 0) / n)
        best = err.argmin(0)
        print(f'minirocket alpha/n per step: {ALPHA_GRID[best].tolist()}', flush=True)
        return grid[best]

    def calibrate(self, cal: Windows) -> None:
        """Band offsets from residuals on ``cal``, windows the fit and alpha never saw."""
        from zoo.export import band_offsets
        res = cal.target - self.predict(cal)
        if (np.isfinite(res).sum(0) < 20).any():
            raise ValueError('too few measured calibration targets for a band')
        self.offsets = band_offsets(res)

    def predict(self, w: Windows) -> np.ndarray:
        return w.bg[:, -1:].astype(np.float64) + self._delta(w, self.coef)

    def predict_quantiles(self, w: Windows) -> np.ndarray:
        if not np.asarray(self.offsets).size:
            raise ValueError('minirocket has no band: calibrate() was not run')
        return self.predict(w)[..., None] + np.asarray(self.offsets).T[None]

    def state(self) -> dict[str, np.ndarray]:
        cpu = {k: getattr(self, k).cpu().numpy() for k in ('mu', 'inv', 'coef')}
        return {**cpu, **{f'k_{k}': v for k, v in self.kernels.items()}, 'alphas': self.alphas,
                'ch_mu': self.ch_mu, 'ch_sd': self.ch_sd, 'offsets': self.offsets}

    def load(self, state: dict[str, np.ndarray]) -> None:
        self.kernels = {k[2:]: v for k, v in state.items() if k.startswith('k_')}
        self.ch_mu, self.ch_sd, self.offsets = state['ch_mu'], state['ch_sd'], state['offsets']
        self.alphas = state['alphas']
        self.mu, self.inv, self.coef = (torch.as_tensor(state[k], device=self.device)
                                        for k in ('mu', 'inv', 'coef'))


MODELS = {'minirocket': MiniRocket}
