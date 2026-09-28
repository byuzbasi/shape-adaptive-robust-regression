"""
shapepen.py
===========

Estimators for "From residual-moment penalties to shape-adaptive robust estimation
under skewness and heavy tails".

1. ShapePenalizedEN minimises

       L(beta) = loss(r) + lam * ( alpha*||beta||_1 + (1-alpha)/2*||beta||_2^2 )
                 + lam1 * phi_eps( m3(v) ) + lam2 * softplus_tau( m4(v) - 3*m2(v)^2 ),

   where r = y - X beta are residuals, v are centred residuals and m_k(v) = mean(v^k).
   * robust=False: loss = (1/2n) sum r_i^2 and v_i = r_i - mean(r).  With lam1 = lam2 = 0
     this is the elastic net (EN); with lam1, lam2 > 0 it is the skewness/kurtosis
     penalised elastic net (SK-EN).
   * robust=True: loss = (1/n) sum s^2 rho_H(r_i / s) (Huber) with a fixed scale s.  With
     lam1 = lam2 = 0 this is the Huber elastic net (H-EN) used as pilot.  (Shape
     penalties can be combined with it, computed from v_i = c*tanh(r_i/(s c)) centred;
     this option is not used in the paper.)
   The absolute value is smoothed by phi_eps(z) = sqrt(z^2 + eps) and the positive part
   by the softplus tau*log(1 + exp(z/tau)).  Optimisation is by cyclic coordinate
   gradient descent: a proximal (soft-thresholding) step from the partial gradient of
   the smooth part, with an Armijo step length on the full objective (Tseng and Yun,
   2009), and continuation in eps when lam1 > 0.  Inner loops are compiled with numba.

2. ShapeAdaptiveEN (SA-EN) estimates, from pilot residuals, the variance-minimising
   score within span{psi_c, psi_c^2, psi_c^3}, psi_c(u) = c*tanh(u/c), forms one-step
   pseudo-responses and fits a cross-validated elastic net to them.

3. QuantileLasso gives the LAD lasso and the composite quantile regression lasso,
   solved exactly by linear programming.
"""

import numpy as np
from numba import njit

EPS_ABS = 1e-6      # smoothing constant for |z|
TAU_POS = 1e-3      # smoothing constant for (z)_+
SKEN_MAX_SWEEPS = 120_000  # per smoothing stage; fits still stop at the same KKT bound
SKEN_OBJECTIVE_TOL = 1e-9
SKEN_KKT_TOL = 1e-3       # final target objective, on the standardized solver scale
EN_FAMILY_MAX_SWEEPS = 30_000  # CV/refit cap when a KKT bound is requested
EN_FAMILY_KKT_TOL = 1e-3       # all EN/H-EN/SA-EN CV candidates and refits
SAEN_KKT_TOL = EN_FAMILY_KKT_TOL


# ---------------------------------------------------------------------------
# elementary smooth surrogates
# ---------------------------------------------------------------------------
@njit(cache=True)
def _softplus(z, tau):
    # numerically stable tau*log(1+exp(z/tau))
    t = z / tau
    if t > 30.0:
        return z
    if t < -30.0:
        return tau * np.exp(t)
    return tau * np.log1p(np.exp(t))


@njit(cache=True)
def _sigmoid(z, tau):
    t = z / tau
    if t >= 0:
        return 1.0 / (1.0 + np.exp(-t))
    e = np.exp(t)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# objective pieces evaluated from a residual vector
# ---------------------------------------------------------------------------
@njit(cache=True)
def _loss_and_shape(r, robust, s, c_huber, c_clip, lam1, lam2, eps):
    """Return loss(r) + shape penalty, plus the moments needed for gradients."""
    n = r.shape[0]
    loss = 0.0
    # shape residuals
    v = np.empty(n)
    mean_u = 0.0
    for i in range(n):
        if robust:
            z = r[i] / s
            az = abs(z)
            if az <= c_huber:
                loss += 0.5 * z * z * s * s
            else:
                loss += (c_huber * az - 0.5 * c_huber * c_huber) * s * s
            u = c_clip * np.tanh(z / c_clip)
        else:
            loss += 0.5 * r[i] * r[i]
            u = r[i]
        v[i] = u
        mean_u += u
    loss /= n
    mean_u /= n
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        v[i] -= mean_u
        v2 = v[i] * v[i]
        m2 += v2
        m3 += v2 * v[i]
        m4 += v2 * v2
    m2 /= n
    m3 /= n
    m4 /= n
    K = m4 - 3.0 * m2 * m2
    pen = lam1 * np.sqrt(m3 * m3 + eps) + lam2 * _softplus(K, TAU_POS)
    return loss + pen, v, m2, m3, K


@njit(cache=True)
def _full_objective(r, beta, pw, lam, alpha, robust, s, c_huber, c_clip, lam1, lam2, eps):
    val, _, _, _, _ = _loss_and_shape(r, robust, s, c_huber, c_clip, lam1, lam2, eps)
    l1 = 0.0
    l2 = 0.0
    for j in range(beta.shape[0]):
        l1 += pw[j] * abs(beta[j])
        l2 += pw[j] * beta[j] * beta[j]
    return val + lam * (alpha * l1 + 0.5 * (1.0 - alpha) * l2)


@njit(cache=True)
def _smooth_part_trial(r, xj, t_step, robust, s, c_huber, c_clip, lam1, lam2, eps):
    """loss + shape penalty at residual r - t_step * x_j (without forming beta)."""
    n = r.shape[0]
    rt = np.empty(n)
    for i in range(n):
        rt[i] = r[i] - t_step * xj[i]
    val, _, _, _, _ = _loss_and_shape(rt, robust, s, c_huber, c_clip, lam1, lam2, eps)
    return val


@njit(cache=True)
def _cgd(X, y, beta0, pw, lam, alpha, lam1, lam2, robust, s, c_huber, c_clip,
         max_sweeps, tol, sigma_armijo, rho_armijo, eps, kkt_tol=0.0):
    n, p = X.shape
    beta = beta0.copy()
    r = y - X @ beta
    # coordinate-wise curvature of the quadratic part: (1/n)||x_j||^2 + ridge
    h = np.empty(p)
    for j in range(p):
        h[j] = max(np.sum(X[:, j] ** 2) / n + pw[j] * lam * (1.0 - alpha), 1e-8)
    h_min = 0.1 * h.copy()
    obj = _full_objective(r, beta, pw, lam, alpha, robust, s, c_huber, c_clip, lam1, lam2, eps)
    n_sweeps = 0
    converged = False
    for sweep in range(max_sweeps):
        n_sweeps = sweep + 1
        obj_old = obj
        for j in range(p):
            xj = X[:, j]
            smooth_val, v, m2, m3, K = _loss_and_shape(r, robust, s, c_huber,
                                                      c_clip, lam1, lam2, eps)
            # ---- partial gradient of the smooth part ----------------------
            # loss part
            g = 0.0
            if robust:
                for i in range(n):
                    z = r[i] / s
                    psi = min(max(z, -c_huber), c_huber)
                    g -= psi * s * xj[i]
            else:
                for i in range(n):
                    g -= r[i] * xj[i]
            g /= n
            g += pw[j] * lam * (1.0 - alpha) * beta[j]
            # shape part: d v_i / d beta_j = -(1/s) (w_i x_ij - mean(w x_j)),
            # with w_i = psi_c'(r_i/s) (w_i = 1, s = 1 for the least-squares version)
            if lam1 > 0.0 or lam2 > 0.0:
                d3 = m3 / np.sqrt(m3 * m3 + eps)
                d4 = _sigmoid(K, TAU_POS)
                # a_i = d(penalty)/d v_i ;  gradient = -(1/s) sum (a_i - abar) w_i x_ij
                abar = 0.0
                a = np.empty(n)
                for i in range(n):
                    vi = v[i]
                    a[i] = (lam1 * d3 * 3.0 * vi * vi
                            + lam2 * d4 * (4.0 * vi * vi * vi - 12.0 * m2 * vi)) / n
                    abar += a[i]
                abar /= n
                gs = 0.0
                sc = s if robust else 1.0
                for i in range(n):
                    if robust:
                        th = np.tanh(r[i] / (s * c_clip))
                        wi = 1.0 - th * th
                    else:
                        wi = 1.0
                    gs += (a[i] - abar) * wi * xj[i]
                g -= gs / sc
            # ---- proximal (soft-thresholding) direction --------------------
            zj = beta[j] - g / h[j]
            thr = pw[j] * lam * alpha / h[j]
            if zj > thr:
                newb = zj - thr
            elif zj < -thr:
                newb = zj + thr
            else:
                newb = 0.0
            d = newb - beta[j]
            if abs(d) < 1e-14:
                continue
            # ---- Armijo rule on the full objective (Tseng & Yun, 2009) -----
            l1_old = pw[j] * lam * alpha * abs(beta[j])
            delta = g * d + pw[j] * lam * alpha * abs(beta[j] + d) - l1_old
            ridge_old = 0.5 * pw[j] * lam * (1.0 - alpha) * beta[j] ** 2
            f_old = smooth_val + l1_old + ridge_old
            t_step = 1.0
            accepted = False
            for _ in range(40):
                bnew = beta[j] + t_step * d
                f_new = (_smooth_part_trial(r, xj, t_step * d, robust, s, c_huber,
                                            c_clip, lam1, lam2, eps)
                         + pw[j] * lam * alpha * abs(bnew)
                         + 0.5 * pw[j] * lam * (1.0 - alpha) * bnew * bnew)
                if f_new <= f_old + sigma_armijo * t_step * delta:
                    accepted = True
                    break
                t_step *= rho_armijo
            # adapt the curvature estimate h_j within fixed bounds: a short accepted
            # step signals that h_j underestimates the local curvature
            if t_step < 1.0:
                h[j] = min(h[j] / t_step, 1e8)
            else:
                h[j] = max(0.8 * h[j], h_min[j])
            if accepted:
                step = t_step * d
                beta[j] += step
                for i in range(n):
                    r[i] -= step * xj[i]
        obj = _full_objective(r, beta, pw, lam, alpha, robust, s, c_huber, c_clip, lam1, lam2, eps)
        if abs(obj_old - obj) < tol * (1.0 + abs(obj_old)):
            if kkt_tol > 0.0:
                # Recompute residuals before testing stationarity to avoid drift.
                r = y - X @ beta
                obj = _full_objective(r, beta, pw, lam, alpha, robust, s,
                                      c_huber, c_clip, lam1, lam2, eps)
                kkt = _kkt_residual(X, r, beta, pw, lam, alpha, lam1, lam2,
                                    robust, s, c_huber, c_clip, eps)
                if not np.isfinite(kkt) or kkt > kkt_tol:
                    continue
            converged = True
            break
    return beta, obj, n_sweeps, converged


@njit(cache=True)
def _kkt_residual(X, r, beta, pw, lam, alpha, lam1, lam2, robust, s,
                  c_huber, c_clip, eps):
    """Maximum composite KKT residual on the solver's standardized scale.

    This read-only diagnostic is also used as an additional stopping condition
    when the caller explicitly supplies a positive kkt_tol.
    """
    n = len(r)
    if robust:
        score = s * np.clip(r / s, -c_huber, c_huber)
    else:
        score = r
    grad = -(X.T @ score) / n + pw * lam * (1.0 - alpha) * beta
    if lam1 > 0.0 or lam2 > 0.0:
        _, v, m2, m3, K = _loss_and_shape(r, robust, s, c_huber,
                                          c_clip, lam1, lam2, eps)
        d3 = m3 / np.sqrt(m3 * m3 + eps)
        d4 = _sigmoid(K, TAU_POS)
        a = (lam1 * d3 * 3.0 * v ** 2
             + lam2 * d4 * (4.0 * v ** 3 - 12.0 * m2 * v)) / n
        w = np.ones_like(r)
        if robust:
            w = 1.0 - np.tanh(r / (s * c_clip)) ** 2
        grad -= X.T @ ((a - a.mean()) * w) / (s if robust else 1.0)
    l1 = pw * lam * alpha
    nonzero = np.abs(beta) > 1e-8
    violation = np.where(nonzero, np.abs(grad + l1 * np.sign(beta)),
                         np.maximum(np.abs(grad) - l1, 0.0))
    return float(np.max(violation))


# ---------------------------------------------------------------------------
# user-facing estimator
# ---------------------------------------------------------------------------
class ShapePenalizedEN:
    """Shape-penalised elastic net (SK-EN) and its robust version (RSK-EN).

    Parameters
    ----------
    lam, alpha : elastic-net level and mixing weight (alpha = 1 gives the lasso)
    lam1, lam2 : weights of the skewness and excess-kurtosis penalties
    robust     : False -> least-squares loss with raw centred residual moments (SK-EN)
                 True  -> Huber loss with moments of clipped standardised residuals (RSK-EN)
    c_huber    : Huber tuning constant (on the standardised residual scale)
    c_clip     : clipping constant for the shape residuals
    scale      : fixed residual scale s for the robust version (None -> MAD of a
                 preliminary Huber fit with lam1 = lam2 = 0)
    kkt_tol    : optional final-target KKT bound in addition to objective change;
                 None preserves the original objective-change stopping rule
    """

    def __init__(self, lam=0.1, alpha=0.5, lam1=0.0, lam2=0.0, robust=False,
                 c_huber=1.345, c_clip=2.5, scale=None, max_sweeps=1000,
                 tol=1e-9, beta_init=None, eps=EPS_ABS, kkt_tol=None):
        self.lam, self.alpha, self.lam1, self.lam2 = lam, alpha, lam1, lam2
        self.robust, self.c_huber, self.c_clip, self.scale = robust, c_huber, c_clip, scale
        self.max_sweeps, self.tol, self.beta_init = max_sweeps, tol, beta_init
        self.eps = eps
        if kkt_tol is not None and (not np.isfinite(kkt_tol) or kkt_tol <= 0):
            raise ValueError("kkt_tol must be positive and finite, or None")
        self.kkt_tol = kkt_tol

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, p = X.shape
        # centre and scale the design, centre and scale the response; the intercept
        # is unpenalised and is recovered at the end
        self.x_mean_ = X.mean(0)
        self.x_sd_ = X.std(0)
        self.x_sd_[self.x_sd_ == 0] = 1.0
        # the last column is the (unpenalised) intercept
        Xs = np.column_stack([(X - self.x_mean_) / self.x_sd_, np.ones(n)])
        pw = np.ones(p + 1)
        pw[p] = 0.0
        self.y_loc_ = np.median(y) if self.robust else y.mean()
        self.y_sd_ = y.std() if y.std() > 0 else 1.0
        ys = (y - self.y_loc_) / self.y_sd_
        b0 = np.zeros(p + 1)
        if self.beta_init is not None:
            b0[:p] = np.asarray(self.beta_init, float)
        s = 1.0
        if self.robust:
            if self.scale is None:
                # preliminary Huber elastic net (lam1 = lam2 = 0) started from a MAD
                # scale of the response; its residual MAD is the fixed scale s
                s0 = 1.4826 * np.median(np.abs(ys - np.median(ys))) + 1e-12
                bpre, _, _, _ = _cgd(Xs, ys, b0, pw, self.lam, self.alpha, 0.0, 0.0, True,
                                  s0, self.c_huber, self.c_clip, self.max_sweeps,
                                  self.tol, 1e-4, 0.5, self.eps)
                rpre = ys - Xs @ bpre
                s = 1.4826 * np.median(np.abs(rpre - np.median(rpre))) + 1e-12
                b0 = bpre
            else:
                s = self.scale / self.y_sd_
        self.s_ = s
        # continuation in the smoothing constant: the fit for a larger eps is used as
        # the warm start for the next smaller one, ending at the target value
        beta, sweeps = b0, 0
        sweeps_by_stage = []
        converged_by_stage = []
        schedule = [self.eps]
        if self.lam1 > 0:
            schedule = [e for e in (1e-2, 1e-3, 1e-4, 1e-5) if e > self.eps] + [self.eps]
        for e in schedule:
            target_kkt_tol = self.kkt_tol if e == self.eps and self.kkt_tol is not None else 0.0
            beta, obj, sw, converged = _cgd(Xs, ys, beta, pw, self.lam, self.alpha, self.lam1,
                                  self.lam2, self.robust, s, self.c_huber, self.c_clip,
                                  self.max_sweeps, self.tol, 1e-4, 0.5, e, target_kkt_tol)
            sweeps += sw
            sweeps_by_stage.append(int(sw))
            converged_by_stage.append(bool(converged))
        self.beta_std_ = beta[:p]
        self.coef_ = beta[:p] * self.y_sd_ / self.x_sd_
        self.intercept_ = self.y_loc_ + self.y_sd_ * beta[p] - self.x_mean_ @ self.coef_
        self.resid_std_ = ys - Xs @ beta
        self.objective_, self.n_sweeps_ = obj, sweeps
        self.fit_diagnostics_ = dict(
            objective=float(obj),
            max_kkt_residual=_kkt_residual(
                Xs, self.resid_std_, beta, pw, self.lam, self.alpha,
                self.lam1, self.lam2, self.robust, s, self.c_huber,
                self.c_clip, self.eps),
            sweeps_by_stage=sweeps_by_stage,
            smoothing_schedule=schedule,
            converged_by_stage=converged_by_stage,
            hit_iteration_cap=not converged_by_stage[-1],
            intermediate_cap_count=sum(not value for value in converged_by_stage[:-1]),
            kkt_tolerance=self.kkt_tol,
        )
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# cross-validation helpers
# ---------------------------------------------------------------------------
def robust_spread(e):
    """Normalised median absolute deviation about the median of a residual vector."""
    return 1.4826 * np.median(np.abs(e - np.median(e)))


def lambda_max(X, y, alpha):
    """Smallest elastic-net level giving the zero solution on the standardised scale."""
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    return np.max(np.abs(Xs.T @ ys)) / (len(y) * max(alpha, 1e-3))


def fold_ids(n, k, seed):
    return np.random.default_rng(seed).permutation(n) % k


def cv_select(fit_predict, X, y, candidates, k=5, seed=0, criterion="mse", target=None,
              diagnostics=None):
    """Generic K-fold CV.

    fit_predict(Xtr, ytr, Xte, cand) -> predictions for Xte.
    Out-of-fold residuals are pooled over folds and scored by the mean squared error
    ('mse') or by the normalised MAD ('mad').  `target` (default y) is the response on
    which models are fitted; predictions are always scored against `target`.
    """
    t = y if target is None else target
    folds = fold_ids(len(y), k, seed)
    oof = np.zeros((len(candidates), len(y)))
    for f in range(k):
        tr, te = folds != f, folds == f
        for ci, cand in enumerate(candidates):
            result = fit_predict(X[tr], t[tr], X[te], cand)
            if isinstance(result, tuple):
                pred, fit_diag = result
                if diagnostics is not None:
                    diagnostics.append(dict(fold=int(f), candidate=int(ci),
                                            fit=fit_diag))
            else:
                pred = result
            oof[ci, te] = t[te] - pred
    if criterion == "mse":
        score = np.mean(oof ** 2, axis=1)
    else:
        score = np.array([robust_spread(e) for e in oof])
    return candidates[int(np.argmin(score))], score


# ---------------------------------------------------------------------------
# elastic-net family fitted with the coordinate gradient descent solver
# ---------------------------------------------------------------------------
def fit_en_cv(X, y, alpha=0.5, n_lam=15, k=5, seed=0, robust=False, criterion=None,
              kkt_tol=None):
    """EN (CV-MSE) or Huber EN (CV-MAD), with an optional CV/refit KKT bound."""
    crit = criterion or ("mad" if robust else "mse")
    grid = list(lambda_max(X, y, alpha) * np.logspace(-3, 0, n_lam))
    cv_sweeps = 500 if kkt_tol is None else EN_FAMILY_MAX_SWEEPS
    refit_sweeps = 1000 if kkt_tol is None else EN_FAMILY_MAX_SWEEPS

    def fp(Xtr, ytr, Xte, lam):
        fit = ShapePenalizedEN(lam=lam, alpha=alpha, robust=robust, max_sweeps=cv_sweeps,
                               tol=1e-7, kkt_tol=kkt_tol).fit(Xtr, ytr)
        return fit.predict(Xte), fit.fit_diagnostics_
    cv_diagnostics = []
    lam, _ = cv_select(fp, X, y, grid, k, seed, crit, diagnostics=cv_diagnostics)
    m = ShapePenalizedEN(lam=lam, alpha=alpha, robust=robust,
                         max_sweeps=refit_sweeps,
                         kkt_tol=kkt_tol).fit(X, y)
    m.cv_choice_ = lam
    m.cv_selected_index_ = grid.index(lam)
    m.cv_diagnostics_ = cv_diagnostics
    m.cv_criterion_ = crit
    return m


def fit_sken_cv(X, y, lam, alpha=0.5, shape_grid=(0.0, 0.01, 0.1, 1.0), k=5, seed=0):
    """Skewness/kurtosis-penalised elastic net: (lam1, lam2) chosen by CV-MSE with the
    elastic-net level fixed at `lam` (the grid contains lam1 = lam2 = 0)."""
    cands = [(a, b) for a in shape_grid for b in shape_grid]

    def fp(Xtr, ytr, Xte, c):
        fit = ShapePenalizedEN(lam=lam, alpha=alpha, lam1=c[0], lam2=c[1],
                               max_sweeps=SKEN_MAX_SWEEPS, tol=SKEN_OBJECTIVE_TOL,
                               kkt_tol=SKEN_KKT_TOL).fit(Xtr, ytr)
        return fit.predict(Xte), fit.fit_diagnostics_
    cv_diagnostics = []
    best, _ = cv_select(fp, X, y, cands, k, seed, "mse", diagnostics=cv_diagnostics)
    m = ShapePenalizedEN(lam=lam, alpha=alpha, lam1=best[0], lam2=best[1],
                         max_sweeps=SKEN_MAX_SWEEPS, tol=SKEN_OBJECTIVE_TOL,
                         kkt_tol=SKEN_KKT_TOL).fit(X, y)
    m.cv_choice_ = best
    m.cv_selected_index_ = cands.index(best)
    m.cv_diagnostics_ = cv_diagnostics
    m.cv_criterion_ = "mse"
    return m


# ---------------------------------------------------------------------------
# shape-adaptive elastic net (SA-EN)
# ---------------------------------------------------------------------------
def shape_basis(u, c):
    """Bounded shape basis b(u) = (psi, psi^2, psi^3), psi(u) = c*tanh(u/c), and b'(u)."""
    psi = c * np.tanh(u / c)
    dpsi = 1.0 - (psi / c) ** 2
    B = np.vstack([psi, psi ** 2, psi ** 3])
    dB = np.vstack([dpsi, 2.0 * psi * dpsi, 3.0 * psi ** 2 * dpsi])
    return B, dB


def optimal_score(u, c):
    """Variance-minimising coefficients a = M^{-1} d within span{b_1, b_2, b_3}.

    Returns (a, efficiency) where efficiency = d' M^{-1} d, so that the estimated
    asymptotic variance factor of the resulting estimator is s^2 / efficiency.
    """
    B, dB = shape_basis(u, c)
    M = np.cov(B)
    d = dB.mean(1)
    a = np.linalg.solve(M + 1e-10 * np.eye(3), d)
    return a, float(d @ a)


class ShapeAdaptiveEN:
    """Shape-adaptive elastic net.

    Starting from a pilot fit, each iteration estimates the variance-minimising
    bounded score g = a' b(u) from the standardised pilot residuals, forms the
    one-step pseudo-responses

        z_i = x_i' beta_pilot + mu + s * (g(u_i) - mean g) / mean g'(u_i),

    and fits a cross-validated elastic net to (X, z). Optional kkt_tol applies
    to the Huber pilot and pseudo-response CV/refit solves.
    """

    def __init__(self, alpha=0.5, c_grid=(1.0, 1.5, 2.0, 3.0, 5.0), n_iter=2,
                 n_lam=15, k=5, seed=0, pilot=None, kkt_tol=None):
        self.alpha, self.c_grid, self.n_iter = alpha, c_grid, n_iter
        self.n_lam, self.k, self.seed, self.pilot = n_lam, k, seed, pilot
        self.kkt_tol = kkt_tol

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        pil = self.pilot if self.pilot is not None else fit_en_cv(
            X, y, self.alpha, self.n_lam, self.k, self.seed,
            robust=True, kkt_tol=self.kkt_tol)
        self.pilot_diagnostics_ = dict(fit=pil.fit_diagnostics_,
                                       cv=pil.cv_diagnostics_,
                                       selected_index=pil.cv_selected_index_)
        coef = pil.coef_.copy()
        self.history_ = []
        for _ in range(self.n_iter):
            r = y - X @ coef
            mu = np.median(r)
            s = robust_spread(r) + 1e-12
            u = (r - mu) / s
            fits = [(optimal_score(u, c), c) for c in self.c_grid]
            (a, eff), c = max(fits, key=lambda t: t[0][1])
            B, dB = shape_basis(u, c)
            g, gp = a @ B, a @ dB
            z = X @ coef + mu + s * (g - g.mean()) / gp.mean()
            en = fit_en_cv(X, z, self.alpha, self.n_lam, self.k, self.seed,
                           robust=False, kkt_tol=self.kkt_tol)
            coef = en.coef_.copy()
            self.history_.append(dict(c=c, a=a, var_factor=s ** 2 / eff, scale=s,
                                      fit_diagnostics=en.fit_diagnostics_,
                                      cv_diagnostics=en.cv_diagnostics_,
                                      cv_selected_index=en.cv_selected_index_))
        self.coef_ = coef
        r = y - X @ coef
        self.intercept_ = np.median(r)
        self.a_, self.c_ = self.history_[-1]["a"], self.history_[-1]["c"]
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def diagnostic_summary(model):
    """Compact numerical diagnostics for one completed estimator fit.

    LP-based quantile fits fail explicitly if the optimizer fails and therefore
    have no coordinate-descent KKT diagnostic.
    """
    if isinstance(model, ShapeAdaptiveEN):
        refits = [model.pilot_diagnostics_["fit"]]
        cv_fits = list(model.pilot_diagnostics_["cv"])
        selected_cv = [d for d in cv_fits
                       if d["candidate"] == model.pilot_diagnostics_["selected_index"]]
        for step in model.history_:
            refits.append(step["fit_diagnostics"])
            cv_fits.extend(step["cv_diagnostics"])
            selected_cv.extend(d for d in step["cv_diagnostics"]
                               if d["candidate"] == step["cv_selected_index"])
    elif isinstance(model, ShapePenalizedEN):
        refits = [model.fit_diagnostics_]
        cv_fits = getattr(model, "cv_diagnostics_", [])
        selected_cv = [d for d in cv_fits
                       if d["candidate"] == getattr(model, "cv_selected_index_", None)]
    else:
        return dict(refit_max_kkt=None, refit_cap_count=0,
                    cv_max_kkt=None, cv_cap_count=0, cv_fit_count=0,
                    selected_cv_max_kkt=None, selected_cv_cap_count=0,
                    refit_intermediate_cap_count=0, cv_intermediate_cap_count=0,
                    selected_cv_intermediate_cap_count=0)
    return dict(
        refit_max_kkt=max(d["max_kkt_residual"] for d in refits),
        refit_cap_count=sum(bool(d["hit_iteration_cap"]) for d in refits),
        cv_max_kkt=(max(d["fit"]["max_kkt_residual"] for d in cv_fits)
                    if cv_fits else None),
        cv_cap_count=sum(bool(d["fit"]["hit_iteration_cap"]) for d in cv_fits),
        cv_fit_count=len(cv_fits),
        selected_cv_max_kkt=(max(d["fit"]["max_kkt_residual"] for d in selected_cv)
                             if selected_cv else None),
        selected_cv_cap_count=sum(bool(d["fit"]["hit_iteration_cap"])
                                  for d in selected_cv),
        refit_intermediate_cap_count=sum(d["intermediate_cap_count"] for d in refits),
        cv_intermediate_cap_count=sum(d["fit"]["intermediate_cap_count"] for d in cv_fits),
        selected_cv_intermediate_cap_count=sum(d["fit"]["intermediate_cap_count"]
                                               for d in selected_cv),
    )


# ---------------------------------------------------------------------------
# composite quantile regression lasso (Zou and Yuan, 2008) and LAD lasso
# ---------------------------------------------------------------------------
def _cqr_lasso_lp(X, y, lam, taus):
    """min (1/n) sum_k sum_i rho_tau_k(y_i - b_k - x_i' beta) + lam * ||beta||_1."""
    from scipy.optimize import linprog
    from scipy import sparse
    n, p = X.shape
    K = len(taus)
    # variables: beta+ (p), beta- (p), b (K, free), u+ (nK), u- (nK)
    nv = 2 * p + K + 2 * n * K
    c = np.concatenate([lam * np.ones(2 * p), np.zeros(K),
                        np.concatenate([np.full(n, t / n) for t in taus]),
                        np.concatenate([np.full(n, (1 - t) / n) for t in taus])])
    Xk = sparse.csr_matrix(np.tile(X, (K, 1)))
    Bk = sparse.kron(sparse.eye(K), np.ones((n, 1)))
    I = sparse.eye(n * K)
    A = sparse.hstack([Xk, -Xk, Bk, I, -I]).tocsr()
    b = np.tile(y, K)
    bounds = [(0, None)] * (2 * p) + [(None, None)] * K + [(0, None)] * (2 * n * K)
    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Quantile linear program failed: {res.message}")
    beta = res.x[:p] - res.x[p:2 * p]
    return beta, res.x[2 * p:2 * p + K]


class QuantileLasso:
    """LAD lasso (taus = (0.5,)) or composite quantile regression lasso."""

    def __init__(self, lam=0.01, taus=(0.5,)):
        self.lam, self.taus = lam, taus

    def fit(self, X, y):
        self.xm_, self.xs_ = X.mean(0), X.std(0)
        Xs = (X - self.xm_) / self.xs_
        beta, b = _cqr_lasso_lp(Xs, y, self.lam, self.taus)
        self.coef_ = beta / self.xs_
        # intercept: median-type location (the middle of the fitted quantile levels)
        self.intercept_ = np.median(b) - self.xm_ @ self.coef_
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def fit_quantile_lasso_cv(X, y, taus=(0.5,), n_lam=10, k=5, seed=0):
    Xs = (X - X.mean(0)) / X.std(0)
    lmax = len(taus) * np.max(np.abs(Xs.T @ np.sign(y - np.median(y)))) / (2 * len(y))
    grid = list(lmax * np.logspace(-3, 0, n_lam))

    def fp(Xtr, ytr, Xte, lam):
        return QuantileLasso(lam, taus).fit(Xtr, ytr).predict(Xte)
    lam, _ = cv_select(fp, X, y, grid, k, seed, "mad")
    m = QuantileLasso(lam, taus).fit(X, y)
    m.cv_choice_ = lam
    return m
