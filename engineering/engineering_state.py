"""Fixed environmental-state integrator fit utilities, without old imports."""
import math, io
from contextlib import redirect_stdout
import numpy as np
from scipy.optimize import lsq_linear
SOURCE="linhai"
UPPER=20.
def state_functions(u, c):
    """Nonnegative bounded rate basis, independent of any corrosion response."""
    u = np.asarray(u, dtype=float)
    assert np.isfinite(u).all() and (u >= 0).all()
    if c['family'] == 'constant':
        return np.ones((*u.shape, 1))
    z = -np.expm1(-u / c['tau'])
    if c['family'] == 'bernstein':
        m = c['degree']
        return np.stack([math.comb(m, j) * z**j * (1-z)**(m-j)
                         for j in range(m+1)], axis=-1)
    if c['family'] == 'protection':
        return np.stack([np.ones_like(u), np.exp(-u/c['tau'])], axis=-1)
    a = (u / c['tau']) ** c['power']
    activation = a / (1 + a)
    if c['family'] == 'activation':
        return np.stack([np.ones_like(u), activation], axis=-1)
    return np.stack([np.ones_like(u), np.exp(-u/c['protection_tau']), activation], axis=-1)


def state_clock(driver, durations):
    step = np.asarray(driver) * np.asarray(durations) / 365.25
    assert np.isfinite(step).all() and (step >= 0).all()
    return np.r_[0., np.cumsum(step)[:-1]], step


def design_for_fit(B, y, months, c):
    n = len(y)
    mean_y = max(float(np.mean(abs(y))), 1.)
    if c['objective'] == 'absolute':
        return B / (np.sqrt(n)*mean_y), y / (np.sqrt(n)*mean_y)
    # All normalizations use this source prefix, never later observations.
    floor = max(.1 * mean_y, 1.)
    dt = np.diff(np.r_[0., months])
    assert (dt > 0).all()
    rates = np.diff(np.r_[0., y]) / dt
    rate_B = np.diff(np.vstack([np.zeros((1, B.shape[1])), B]), axis=0) / dt[:, None]
    mean_rate = max(float(np.mean(abs(rates))), 1.)
    return (np.vstack([np.sqrt(.50)*B/(np.sqrt(n)*mean_y),
                       np.sqrt(.25)*B/(np.sqrt(n)*np.maximum(abs(y), floor))[:, None],
                       np.sqrt(.25)*rate_B/(np.sqrt(n)*mean_rate)]),
            np.r_[np.sqrt(.50)*y/(np.sqrt(n)*mean_y),
                  np.sqrt(.25)*y/(np.sqrt(n)*np.maximum(abs(y), floor)),
                  np.sqrt(.25)*rates/(np.sqrt(n)*mean_rate)])


def penalty(c):
    if c['family'] == 'constant':
        return np.zeros((0, c['k']))
    if c['family'] == 'bernstein':
        return np.diff(np.eye(c['k']), axis=0) / 5.
    return np.eye(c['k'])[1:] / 5.


def fit(model, c, y, months, n):
    if n < c['k'] + 1:
        return dict(eligible=False, reason='insufficient_nominal_linear_residual_dof', n=n)
    B = model.basis(SOURCE, c)[:n]
    A, b = design_for_fit(B, y[:n], months[:n], c)
    P = penalty(c)
    augmented = np.vstack([A, np.sqrt(c['lam'])*P])
    response = np.r_[b, np.zeros(len(P))]
    solver_log = io.StringIO()
    with redirect_stdout(solver_log):
        solution = lsq_linear(augmented, response, bounds=(0., UPPER),
                              method='bvls', tol=1e-11, max_iter=200, verbose=2)
    normalized = B/np.maximum(np.linalg.norm(B, axis=0), 1e-30)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(singular > max(1e-12, singular[0]*1e-7)))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-14 else None
    coefficients = solution.x
    prediction = model.basis(SOURCE, c) @ coefficients
    eligible = bool(solution.success and rank == c['k'] and condition is not None and condition < 1e5)
    return dict(eligible=eligible, success=bool(solution.success), n=n,
        coefficients=coefficients.tolist(), amplitude_count=c['k'], active_amplitudes=int(np.sum(coefficients > 1e-7)),
        nominal_linear_residual_dof=n-c['k'], source_SSE_um2=float(np.sum((prediction[:n]-y[:n])**2)),
        cost=float(solution.cost), optimality=float(solution.optimality), iterations=int(solution.nit),
        active_mask=solution.active_mask.tolist(), normalized_design_rank=rank,
        normalized_design_condition=condition, singular_values=singular.tolist(),
        solver_trace=solver_log.getvalue(), source_prediction=prediction.tolist(),
        fitting_design=augmented.tolist(), fitting_response=response.tolist(),
        state_hyperparameters_selected_from_grid={k: c[k] for k in ['driver', 'tau', 'degree', 'power'] if k in c},
        residual_dof_does_not_account_for_state_hyperparameter_selection=True)
