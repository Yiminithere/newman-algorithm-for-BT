"""
utils.py
========
Common utilities for fitting and analyzing the Bradley-Terry (BT) model via
Newman's alpha-scheme, under both synchronous ("full") and asynchronous
("none", i.e. Gauss-Seidel-style) fixed-point updates.

This module consolidates the functions that are duplicated verbatim across
the various experiment notebooks (e.g. example_monkey.ipynb,
examples_cycle.ipynb, the SAT12/ASSISTments/MovieLens pipelines, and the
synthetic SBM convergence-rate studies), so that each notebook can simply do:

    from utils import *

instead of redefining these functions in every notebook.

Notation used throughout
-------------------------
- pi          : the (unnormalized) BT strength vector, one entry per object/player.
- W           : an (n x n) win-count matrix; W[i, j] = number of times i beat j.
- win_list[i]  = [array of opponents i beat, array of corresponding win counts]
- loss_list[i] = [array of opponents i lost to, array of corresponding loss counts]
- alpha       : Newman's mixing parameter. alpha=0 recovers the classical
                Zermelo/Bradley-Terry MM update; alpha=1 recovers a
                different fixed point (sometimes called Newman's algorithm
                in the "alpha-scheme" family used throughout this project).
- sync        : 'full' = synchronous (Jacobi-style) update, all pi_i updated
                simultaneously from the same previous iterate;
                'none' = asynchronous (Gauss-Seidel-style) update, each pi_i
                updated in place using already-updated neighbors within the
                same sweep.
- rho         : the empirically observed linear convergence rate (from
                simulating the iteration and measuring the ratio of
                successive errors).
- rho_bar     : the theoretically predicted convergence rate (from the
                spectral radius / spectral gap of the linearized update
                Jacobian at the fixed point).
"""

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: Imports
# ─────────────────────────────────────────────────────────────────────────────
# numpy      - all core numerical linear algebra (matrix ops, eigenvalues,
#              random sampling) used throughout every function below.
# pandas     - only needed by the notebooks that load real datasets (e.g.
#              CSV files for vervet monkeys, ATP tennis, ASSISTments);
#              kept here so a single `from utils import *` covers both the
#              data-loading notebooks and the synthetic-experiment notebooks.
# random     - Python's built-in RNG; imported for compatibility with legacy
#              code that may call random.* directly (not used by the core
#              functions below, which rely on numpy's RNG instead).
# joblib.Parallel, joblib.delayed
#            - used by get_slope_average() to run multiple independent
#              convergence-rate simulations (different random initial
#              points/seeds) in parallel across CPU cores, since each
#              simulation is embarrassingly parallel and can be slow at
#              high precision (small `tol`, large `maxiter`).
# scipy.linalg.solve_triangular
#            - used to solve the triangular system (I - L) x = U y that
#              arises when analyzing the asynchronous (Gauss-Seidel-style)
#              update's linearization H = (I - L)^{-1} U, where L and U are
#              the strictly-lower and upper (plus diagonal) parts of the
#              Jacobian J. This avoids explicitly forming (I - L)^{-1}.
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from scipy.linalg import solve_triangular


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: Core functions
# ─────────────────────────────────────────────────────────────────────────────

def centering(x):
    """
    Re-center a strictly-positive strength vector x in log-space so that
    the geometric mean of its entries is 1.

    The BT/Newman fixed-point iterations are only identified up to a global
    positive scalar multiple (multiplying all pi_i by the same constant does
    not change any win probability pi_i/(pi_i+pi_j)). To make iterates
    comparable across updates (and to prevent pi from drifting to numerical
    over/underflow), we repeatedly re-center by subtracting the mean of
    log(x) and exponentiating back.

    Parameters
    ----------
    x : array_like, strictly positive
        Unnormalized strength vector.

    Returns
    -------
    array
        Rescaled vector with geometric mean 1.
    """
    logx = np.log(x)
    logx -= np.mean(logx)
    return np.exp(logx)


def get_jacobian(W, pi, alpha, full=False):
    """
    Compute the Jacobian J of Newman's alpha-scheme fixed-point map at the
    point pi, using a closed-form vectorized expression (fast, O(n^2)).

    This is the "convenient" symmetrized/reweighted Jacobian representation
    used for the SYNCHRONOUS spectral-gap analysis (see get_spectral_gap):
    after a similarity transform by diag(bl) (returned when full=True), J
    becomes similar to a matrix whose symmetric part's eigenvalues bound the
    convergence rate.

    Parameters
    ----------
    W : (n, n) array
        Win-count matrix; W[i, j] = number of times i beat j.
    pi : (n,) array
        Point (typically the MLE) at which to linearize the update map.
    alpha : float
        Newman's mixing parameter.
    full : bool, default False
        If True, additionally return the vector `bl` such that the
        similarity-transformed matrix J / bl[None,:] * bl[:,None] is used
        for the symmetrized spectral-gap computation in get_spectral_gap().

    Returns
    -------
    J : (n, n) array
        The Jacobian of the synchronous update map at pi.
    bl : (n,) array, only if full=True
        Rescaling vector sqrt(total_weight / pi) used to symmetrize J.
    """
    # core part: M is the symmetrized total-comparison-count matrix,
    # R = 1/(pi_i+pi_j)^2 is the BT curvature term, K = M*R combines them,
    # and Q = K * pi_i is the (i, j) numerator entry of the Jacobian before
    # normalization.
    M = W + W.T
    R = 1 / (pi[:, None] + pi[None, :])**2
    K = M * R
    Q = K * pi[:, None]

    # normalization: total_weight[i] is the denominator that appears in
    # Newman's update rule for player i (a convex combination of the
    # alpha-weighted win term and the loss term).
    term_alpha = alpha * pi * K.sum(axis=1)
    term_pi_l = (K * pi[None, :]).sum(axis=1)
    total_weight = term_alpha + term_pi_l

    # combining: off-diagonal entries are Q rescaled by 1/total_weight;
    # the diagonal entry captures the self-dependence introduced by the
    # alpha term (Newman's scheme interpolates between two limiting
    # updates, and alpha > 0 makes each pi_i depend partly on itself).
    J = Q / total_weight[:, None]
    diag_val = alpha * Q.sum(axis=1) / total_weight
    np.fill_diagonal(J, 0)
    J += np.diag(diag_val)

    if full == True:
        return J, np.sqrt(total_weight/pi)
    else:
        return J


def get_original_J(W, pi, alpha=0):
    """
    Compute the Jacobian J of Newman's alpha-scheme fixed-point map at pi
    using the original (non-vectorized, entrywise) formula.

    This is mathematically equivalent to get_jacobian(..., full=False) but
    derived and implemented independently (double loop over i, j) as a
    slower, more transparent reference implementation -- useful for sanity
    checking get_jacobian(), and used directly by get_lcf() to compute the
    exact local convergence factor rho (as opposed to the symmetrized
    rho_bar approximation from get_spectral_gap/get_spectral_gap_gs).

    Parameters
    ----------
    W : (n, n) array
        Win-count matrix; W[i, j] = number of times i beat j.
    pi : (n,) array
        Point (typically the MLE) at which to linearize the update map.
    alpha : float, default 0
        Newman's mixing parameter.

    Returns
    -------
    J : (n, n) array
        The Jacobian of the synchronous update map at pi.

    Notes
    -----
    This implementation is O(n^2) per outer loop iteration (so O(n^3)
    overall due to the double loop), which becomes a bottleneck for large n
    (e.g. n ~ 10,000). For large-scale experiments, prefer computing only
    the eigenvalues actually needed via a randomized/iterative method (see
    the Krylov-subspace version of get_lcf discussed in the accompanying
    experiments, using scipy.sparse.linalg.eigs with a LinearOperator that
    applies J or H = (I-L)^{-1}U via matrix-vector products rather than
    forming the dense matrix and calling a full dense eigensolver).
    """
    n = len(pi)
    J = np.zeros((n, n))

    for i in range(n):
        # Common sums reused across all j for row i.
        denom = np.sum((alpha * W[i, :] + W[:, i]) / (pi[i] + pi))
        denom_sq = denom**2

        S1 = np.sum(
            W[i, :] * (alpha * pi[i] + pi) / (pi[i] + pi)
        )

        S2 = np.sum(
            (alpha * W[i, :] + W[:, i]) / (pi[i] + pi)**2
        )

        for j in range(n):
            if j != i:
                term1 = (
                    W[i, j] * pi[i] * (1 - alpha)
                    / (pi[i] + pi[j])**2
                ) / denom

                term2 = (
                    S1
                    * (alpha * W[i, j] + W[j, i])
                    / (pi[i] + pi[j])**2
                ) / denom_sq

                J[i, j] = term1 + term2

            else:
                term1 = np.sum(
                    W[i, :] * pi * (alpha - 1)
                    / (pi[i] + pi)**2
                ) / denom

                term2 = (S1 * S2) / denom_sq

                J[i, i] = term1 + term2
    return J


def get_lcf(W, pi, alpha=0, sync='full'):
    """
    Compute the (exact, empirical) Local Convergence Factor rho: the
    magnitude of the dominant eigenvalue of the update-map Jacobian,
    excluding the trivial eigenvalue 1.

    NOTE: this removes exactly the ONE eigenvalue closest to 1 (the true
    trivial eigenvalue), rather than using np.isclose(eigvals, 1.0) with a
    fixed tolerance. On real data the trivial eigenvalue can drift further
    from exactly 1.0 than np.isclose's default tolerance (rtol=1e-5) --
    e.g. landing at 0.9998 -- in which case np.isclose fails to exclude it,
    and the (spurious) trivial eigenvalue gets reported as rho instead of
    the true second eigenvalue. Removing the single nearest-to-1 eigenvalue
    is robust to this regardless of how much floating-point drift it has,
    as confirmed on the vervet monkey dataset (np.isclose gave rho ~ 0.9998
    for all four alpha/sync combinations, an obvious artifact, while this
    version correctly recovers rho ~ 0.66/0.46/0.96/0.96 matching rho_bar).

    Parameters
    ----------
    W : (n, n) array
        Win-count matrix.
    pi : (n,) array
        Point (typically the MLE) at which to linearize.
    alpha : float, default 0
        Newman's mixing parameter.
    sync : {'full', 'none'}, default 'full'
        'full' -> use the raw synchronous Jacobian J directly.
        'none' -> use the asynchronous (Gauss-Seidel) iteration matrix
                  H = (I - L)^{-1} U.

    Returns
    -------
    float
        rho = |lambda|, the magnitude of the largest-magnitude eigenvalue
        of J (or H) after removing the single eigenvalue nearest to 1.
    """
    J = get_original_J(W, pi, alpha)
    if sync == 'full':
        eigvals = np.linalg.eigvals(J)
    else:
        n = J.shape[0]
        L = np.tril(J, k=-1)
        U = J - L
        I_minus_L = np.eye(n) - L
        H = solve_triangular(I_minus_L, U, lower=True)
        eigvals = np.linalg.eigvals(H)

    idx_trivial = np.argmin(np.abs(eigvals - 1.0))
    filtered = np.delete(eigvals, idx_trivial)
    largest = filtered[np.argmax(np.abs(filtered))]
    return abs(largest)


def slope_from_y(y):
    """
    Ordinary-least-squares slope of y against its index (1, 2, ..., len(y)).

    Used as a simple linear-trend estimator, e.g. for fitting a slope to a
    sequence of log-errors when eyeballing linear convergence on a
    semilog plot.

    Parameters
    ----------
    y : array_like

    Returns
    -------
    float
        OLS slope beta1 of y on x = 1..len(y).
    """
    y = np.array(y)
    n = len(y)
    x = np.arange(1, n+1)
    beta1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
    return beta1


def get_spectral_gap(W, pi, alpha=0):
    """
    Compute the theoretical spectral gap for the SYNCHRONOUS update,
    using the symmetrized Jacobian.

    The raw Jacobian J from get_jacobian() is generally non-symmetric, but
    after the similarity transform J -> J / bl[None,:] * bl[:,None] (using
    the rescaling vector bl returned by get_jacobian(..., full=True)), the
    symmetric part S = (JJ + JJ.T)/2 has real eigenvalues that can be used
    to bound the true (possibly complex) spectral radius of J. The gap
    between the two largest-magnitude eigenvalues of S is used as a
    theoretical proxy for the true convergence gap 1 - rho.

    Parameters
    ----------
    W : (n, n) array
        Win-count matrix.
    pi : (n,) array
        Point (typically the MLE) at which to linearize.
    alpha : float, default 0
        Newman's mixing parameter.

    Returns
    -------
    gap : float
        Difference between the largest and second-largest |eigenvalue|
        of the symmetrized matrix S.
    eigvals_desc : (n,) array
        All eigenvalues of S, sorted in descending order (note: this is
        eigvals[::-1], i.e. sorted by *value* not by |value|; the largest
        eigenvalue of a matrix similar to a stochastic-like Jacobian is
        expected to be 1).
    """
    J, bl = get_jacobian(W, pi, alpha, full=True)
    JJ = J/bl[None, :]*bl[:, None]
    S = (JJ + JJ.T)/2
    eigvals = np.linalg.eigvalsh(S)
    eigvals_sorted = np.sort(abs(eigvals))[::-1]
    gap = eigvals_sorted[0] - eigvals_sorted[1]
    return gap, eigvals[::-1]


def get_spectral_gap_gs(W, pi, alpha=0):
    """
    Compute the theoretical spectral gap for the ASYNCHRONOUS
    (Gauss-Seidel-style) update.

    Forms the Gauss-Seidel iteration matrix H = (I - L)^{-1} U (L =
    strictly lower part of J, U = upper+diagonal part of J) and returns
    the gap between the two largest-magnitude eigenvalues of H directly
    (H need not be symmetric, so eigenvalues can be complex; we sort by
    magnitude rather than symmetrizing).

    Parameters
    ----------
    W : (n, n) array
        Win-count matrix.
    pi : (n,) array
        Point (typically the MLE) at which to linearize.
    alpha : float, default 0
        Newman's mixing parameter.

    Returns
    -------
    gap : float
        |lambda_1| - |lambda_2|, the gap between the two largest-magnitude
        eigenvalues of H.
    eigvals_sorted : (n,) array (complex)
        Eigenvalues of H sorted by descending magnitude.
    """
    J = get_jacobian(W, pi, alpha)
    n = J.shape[0]
    L = np.tril(J, k=-1)
    U = J - L
    I_minus_L = np.eye(n) - L
    H = solve_triangular(I_minus_L, U, lower=True)

    eigvals = np.linalg.eigvals(H)
    eigvals_abs = np.abs(eigvals)
    idx = np.argsort(eigvals_abs)[::-1]
    eigvals_sorted = eigvals[idx]
    gap = np.abs(eigvals_sorted[0]) - np.abs(eigvals_sorted[1])

    return gap, eigvals_sorted


def get_slope(win_list, loss_list, mle, alpha, sync='full',
              tol=1e-12, maxiter=1000, seed=2026):
    """
    Empirically estimate the observed linear convergence rate rho by
    running Newman's alpha-scheme from a single random initialization and
    measuring the average ratio of consecutive errors ||pi_k - mle|| once
    the iteration enters its asymptotic (linear-convergence) regime.

    The iteration is deemed to have entered the linear regime at the first
    iterate j where the error first drops below 1e-3; the returned slope
    is the average of err[k+1]/err[k] over all iterations after j.

    Parameters
    ----------
    win_list, loss_list : dict
        As constructed elsewhere (see get_data), win_list[i] = [opponent
        indices, win counts] and similarly for loss_list.
    mle : (n,) array
        The fixed point (MLE) to converge to; used only to measure error.
    alpha : float
        Newman's mixing parameter.
    sync : {'full', 'none'}, default 'full'
    tol : float, default 1e-12
        Stop the simulated iteration once the error falls below this.
    maxiter : int, default 1000
        Maximum number of iterations to simulate.
    seed : int, default 2026
        Random seed for the initial point pi (drawn lognormal).

    Returns
    -------
    float
        Estimated empirical convergence rate rho (mean ratio of
        consecutive errors post-transient). Returns np.nan if the
        iteration never enters the linear-convergence regime within
        maxiter -- e.g. because the true rho >= 1 (divergent or
        non-convergent case) and the error never drops below 1e-3, or
        because it only crosses that threshold on the very last iterate
        (leaving no subsequent points to estimate a ratio from).
    """
    n = len(win_list)
    np.random.seed(seed)
    pi = np.random.lognormal(0, 1, n)
    err = np.inf
    err_list = []
    i = 0
    j = None   # iterate index where the error first drops below 1e-3
    while err > tol and i < maxiter:
        i += 1
        pi_new = newman_update(pi, win_list, loss_list, alpha=alpha, sync=sync)
        err = np.linalg.norm(mle-pi, ord=2)
        # Guard err_list[-1] against the empty-list case (i == 1): there is
        # no previous error to compare against yet, so the transition check
        # only makes sense from the second iterate onward.
        if err < 1e-3 and (len(err_list) == 0 or err_list[-1] >= 1e-3):
            j = i
        err_list.append(err)
        pi = pi_new.copy()

    if j is None:
        # Never entered the linear-convergence regime within maxiter --
        # typically because rho >= 1 for this (alpha, sync) combination
        # (divergent or non-convergent case). No meaningful slope exists.
        return np.nan

    err_list = np.array(err_list)
    if j >= len(err_list) - 1:
        # Crossed the threshold only at (or after) the last simulated
        # iterate; there are no subsequent points to form a ratio from.
        return np.nan

    return np.mean(err_list[(j+1):]/err_list[j:-1])


def get_slope_average(win_list, loss_list, mle, alpha, sync='full',
                       tol=1e-12, maxiter=1000, N=10):
    """
    Run get_slope() N times in parallel (different random seeds/initial
    points) and return the mean estimated convergence rate along with an
    approximate 95% confidence interval (mean +/- 1.96 * sample std).

    Parameters
    ----------
    win_list, loss_list, mle, alpha, sync, tol, maxiter :
        Passed through to get_slope(); see get_slope() docstring.
    N : int, default 10
        Number of independent random-seed replications.

    Returns
    -------
    m : float
        Mean estimated rho across the N replications (nan-aware: any
        individual replication returning np.nan from get_slope, e.g.
        because that seed's trajectory never entered the linear-
        convergence regime, is excluded rather than propagating nan into
        the overall mean/std). Returns np.nan if ALL replications are nan.
    lo : float
        Approximate lower 95% CI bound.
    hi : float
        Approximate upper 95% CI bound.
    """
    res = np.array(Parallel(n_jobs=-2)(
        delayed(get_slope)(win_list, loss_list, mle, alpha, sync=sync,
                            tol=tol, maxiter=maxiter, seed=2026 + i)
        for i in range(N)
    ))
    if np.all(np.isnan(res)):
        # Every replication failed to enter the linear-convergence regime
        # (e.g. rho >= 1 for this alpha/sync combination) -- no meaningful
        # slope estimate exists.
        return np.nan, np.nan, np.nan
    sd = np.nanstd(res)
    m = np.nanmean(res)
    return m, m - 1.96*sd, m + 1.96*sd


def newman_update_single(i, pi, win_index, win_count, loss_index, loss_count, alpha):
    """
    Compute the updated strength pi_i for a single object i under
    Newman's alpha-scheme, given the current strength vector pi and i's
    win/loss opponents and counts.

    This is the atomic building block used by both the synchronous and
    asynchronous update routines below.

    Parameters
    ----------
    i : int
        Index of the object being updated.
    pi : (n,) array
        Current (full) strength vector (only entries at win_index /
        loss_index are actually read).
    win_index : array of int
        Indices of opponents that i has beaten.
    win_count : array of int/float
        Number of times i beat each corresponding opponent in win_index.
    loss_index : array of int
        Indices of opponents that i has lost to.
    loss_count : array of int/float
        Number of times i lost to each corresponding opponent in
        loss_index.
    alpha : float
        Newman's mixing parameter.

    Returns
    -------
    float
        The new (unnormalized) value of pi_i.
    """
    num = np.sum(win_count*(alpha*pi[i] + pi[win_index])/(pi[i] + pi[win_index]))
    dem = np.sum(alpha*win_count/(pi[i] + pi[win_index])) + np.sum(loss_count/(pi[i] + pi[loss_index]))
    return num/dem


def newman_update_sync(pi, win_list, loss_list, alpha=0, index=None):
    """
    One SYNCHRONOUS (Jacobi-style) sweep of Newman's alpha-scheme: every
    pi_i (for i in `index`) is recomputed from the *same* previous iterate
    pi, then all updates are applied simultaneously and the result is
    re-centered.

    Parameters
    ----------
    pi : (n,) array
        Current strength vector.
    win_list, loss_list : dict
        Per-object win/loss opponent indices and counts.
    alpha : float, default 0
    index : array_like of int, optional
        Subset of indices to update (default: all n indices).

    Returns
    -------
    (n,) array
        The updated, re-centered strength vector.
    """
    pi = np.copy(pi)
    n = len(pi)
    if index is None:
        index = np.arange(n)
    pi[index] = np.array([
        newman_update_single(i, pi, win_list[i][0], win_list[i][1],
                              loss_list[i][0], loss_list[i][1], alpha)
        for i in index
    ])
    return centering(pi)


def newman_update_async(pi, win_list, loss_list, alpha=0, w=1, index=None):
    """
    One ASYNCHRONOUS (Gauss-Seidel-style) sweep of Newman's alpha-scheme:
    each pi_i (for i in `index`, in order) is updated in place, so later
    updates within the same sweep immediately see earlier updates from
    the same sweep. Optionally under-relaxed via the weight w (w=1 is the
    standard full-step async update; w<1 mixes the new value with the
    old one).

    Parameters
    ----------
    pi : (n,) array
        Current strength vector.
    win_list, loss_list : dict
    alpha : float, default 0
    w : float, default 1
        Relaxation weight; new pi_i = w * (Newman update) + (1-w) * old pi_i.
    index : array_like of int, optional
        Order in which to update indices (default: 0, 1, ..., n-1).

    Returns
    -------
    (n,) array
        The updated, re-centered strength vector.
    """
    pi = np.copy(pi)
    n = len(pi)
    if index is None:
        index = np.arange(n)
    for i in index:
        pi[i] = w*newman_update_single(i, pi, win_list[i][0], win_list[i][1],
                                        loss_list[i][0], loss_list[i][1], alpha) \
                + (1-w)*pi[i]
    return centering(pi)


def newman_update(pi, win_list, loss_list, alpha=0, w=1, sync='full', index=None):
    """
    Dispatch to either the synchronous or asynchronous Newman-update sweep.

    Parameters
    ----------
    pi : (n,) array
    win_list, loss_list : dict
    alpha : float, default 0
    w : float, default 1
        Relaxation weight, only used when sync='none'.
    sync : {'full', 'none'}, default 'full'
        'full' -> synchronous (Jacobi) update via newman_update_sync.
        'none' -> asynchronous (Gauss-Seidel) update via newman_update_async.
    index : array_like of int, optional
        Subset/order of indices to update.

    Returns
    -------
    (n,) array
        Updated strength vector.
    """
    if sync == 'full':
        return newman_update_sync(pi, win_list, loss_list, alpha=alpha, index=index)
    elif sync == 'none':
        return newman_update_async(pi, win_list, loss_list, alpha=alpha, w=w, index=index)
    else:
        return 'sync not compatible!'


def newman_fpi(win_list, loss_list, alpha=0, sync='full',
                maxiter=5000, tol=1e-16, err_ord=2, verbose=False):
    """
    Run Newman's alpha-scheme fixed-point iteration to convergence,
    starting from the all-ones vector, to obtain the maximum-likelihood
    estimate (MLE) of the BT strengths.

    Parameters
    ----------
    win_list, loss_list : dict
    alpha : float, default 0
    sync : {'full', 'none'}, default 'full'
    maxiter : int, default 1000
    tol : float, default 1e-16
        Stop once ||pi_new - pi||_{err_ord} < tol.
    err_ord : int, default 2
        Norm order used for the stopping criterion.
    verbose : bool, default False
        If True, print the iterate-to-iterate relative error at every step (useful
        for monitoring convergence / non-convergence in real time, e.g.
        when diagnosing a non-convergent sync/alpha=0 case). Default is
        False so that calling this function inside a large sweep (e.g.
        over many sigma/L/alpha combinations) does not flood the output
        with per-iteration error logs.

    Returns
    -------
    (n,) array
        The converged (or maxiter-truncated) strength vector.
    """
    n = len(win_list)
    pi = np.ones(n)
    err = np.inf
    i = 0
    while err > tol and i < maxiter:
        i += 1
        pi_new = newman_update(pi, win_list, loss_list, alpha=alpha, sync=sync)
        err = np.linalg.norm(pi_new-pi, ord=err_ord)/np.linalg.norm(pi_new, ord=err_ord)
        if verbose:
            print(f"l{err_ord} error:", err)
        pi = pi_new.copy()
    return pi


def sbm_2block(N, p, q, seed=None):
    """
    Generate a symmetric adjacency matrix from a 2-block Stochastic Block
    Model (SBM): N nodes split into two equal-size blocks, with
    within-block edge probability p and cross-block (between-block) edge
    probability q.

    Special cases used throughout the experiments:
      - p == q            -> homogeneous/Erdos-Renyi-like comparison graph.
      - p >> q             -> clustered (two well-separated communities).
      - p == 0, q > 0       -> exactly bipartite comparison graph.

    Parameters
    ----------
    N : int
        Total number of nodes (should be even; split into two blocks of
        size N//2 each).
    p : float
        Within-block edge probability.
    q : float
        Cross-block (between-block) edge probability.
    seed : int, optional
        Seed for the random generator.

    Returns
    -------
    (N, N) int array
        Symmetric 0/1 adjacency matrix (no self-loops).
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=int)
    n = int(N/2)

    # within-block edges
    A[:n, :n] = rng.random((n, n)) < p
    A[n:, n:] = rng.random((n, n)) < p

    # cross-block edges
    A[:n, n:] = rng.random((n, n)) < q

    # keep upper triangle, symmetrize, remove diagonal
    A = np.triu(A, k=1)
    A = A + A.T
    return A


def get_data(A, L, gamma_true, sync='none'):
    """
    Simulate pairwise-comparison data on a given comparison graph A under
    the true BT strengths gamma_true, then fit the MLE.

    For every edge (i, j) present in A (i < j), simulates L
    Binomial(L, gamma_true[i]/(gamma_true[i]+gamma_true[j])) comparisons
    between i and j, recording the win counts in both directions in W.
    Then builds win_list/loss_list from W and computes the MLE via
    newman_fpi (using alpha=0, i.e. the classical Zermelo/BT update, as
    the reference MLE regardless of which alpha/sync combination will
    later be analyzed for convergence rate).

    Parameters
    ----------
    A : (n, n) array
        Symmetric 0/1 (or otherwise binary) adjacency matrix indicating
        which pairs are compared at all.
    L : int
        Number of comparisons simulated per edge.
    gamma_true : (n,) array
        Ground-truth BT strengths used to generate the synthetic data.
    sync : {'full', 'none'}, default 'none'
        Which update scheme to use when computing the MLE via newman_fpi.

    Returns
    -------
    W : (n, n) array
        Simulated win-count matrix.
    win_list, loss_list : dict
        Per-object win/loss opponent indices and counts, built from W.
    mle : (n,) array
        The fitted BT strength vector (alpha=0 MLE).

    Note
    ----
    n is inferred from A.shape[0]; this fixes a bug present in one of the
    original notebook copies of this function, where `n` was read from an
    enclosing global scope instead of being derived from `A`.
    """
    n = A.shape[0]
    W = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i, n):
            if A[i, j] > 0:
                thred = gamma_true[i]/(gamma_true[i]+gamma_true[j])
                W[i, j] = np.random.binomial(L, thred)
                W[j, i] = L - W[i, j]
    win_list = {i: [np.where(W[i] > 0)[0], W[i][np.where(W[i] > 0)[0]]] for i in range(n)}
    loss_list = {i: [np.where(W[:, i] > 0)[0], W[:, i][np.where(W[:, i] > 0)[0]]] for i in range(n)}
    mle = newman_fpi(win_list, loss_list, alpha=0, sync=sync)
    return W, win_list, loss_list, mle


def get_gap(W, mle, alpha, sync):
    """
    Convenience dispatcher: compute the theoretical spectral gap (used to
    derive rho_bar = 1 - gap) for either the synchronous or asynchronous
    update, at the given alpha.

    Parameters
    ----------
    W : (n, n) array
        Win-count matrix.
    mle : (n,) array
        Point (typically the MLE) at which to linearize.
    alpha : float
        Newman's mixing parameter.
    sync : {'full', 'none'}
        'full' -> get_spectral_gap (symmetrized synchronous analysis).
        'none' -> get_spectral_gap_gs (Gauss-Seidel/async analysis).

    Returns
    -------
    float
        The spectral gap (first element of the tuple returned by the
        underlying get_spectral_gap / get_spectral_gap_gs function).
    """
    if sync == 'full':
        return get_spectral_gap(W, mle, alpha=alpha)[0]
    else:
        return get_spectral_gap_gs(W, mle, alpha=alpha)[0]
