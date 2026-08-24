#########################################################################
##   This file is part of the alpha-beta-CROWN (alpha-beta-CROWN)     ##
##   verifier.                                                         ##
##                                                                     ##
##   Copyright (C) 2021-2026 The alpha-beta-CROWN Team                ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Bridge between alpha-beta-CROWN LP models and mlxPDLP (PDHG LP solver).

mlxPDLP: a Primal-Dual Hybrid Gradient LP solver on Apple MLX devices.
- device="cpu"   -> float64 arithmetic throughout (high-accuracy path).
- device="gpu"   -> float32 Metal CSR SpMV (fast path). Portable accuracy
                     ~1e-4; mlxPDLP >= 2026-08-23 supports an opt-in,
                     audited 1e-5 on well-conditioned LPs via bounded
                     host-FP64 correction (see metal_polish_enabled).

Both gurobipy and mlxpdlp are imported lazily, so this module can be
imported in environments that have neither (e.g., plain CI machines).

Soundness contract (important for verification):
  A PDLP iterate is only an *approximation*. This module therefore never
  returns raw objectives as certified bounds. Callers must use
  `audit_certificate()`: it sign-projects and re-audits the returned
  (dual_solution, reduced_cost) in host float64 on the ORIGINAL (unscaled,
  un-presolved) model, and reconstructs the dual objective from the model
  bounds. A nonzero stationarity residual is only certifiable when the
  variable box is finite and the residual correction can be bounded.
  Primal "counterexample" claims must pass `primal_violation()`.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# LP snapshot: everything mlxPDLP needs, with no solver dependency.
# ---------------------------------------------------------------------------

@dataclass
class LPProblem:
    """LP in the form mlxPDLP accepts:
        min  objective' x + objective_constant
        s.t. constraint_lower_bounds <= A x <= constraint_upper_bounds
             variable_lower_bounds <= x <= variable_upper_bounds
    """
    num_variables: int
    num_constraints: int
    row_ptr: np.ndarray            # int32/int64, size num_constraints + 1
    col_indices: np.ndarray        # int32/int64, size nnz
    values: np.ndarray             # float64, size nnz
    objective: np.ndarray          # float64, size num_variables
    objective_constant: float = 0.0
    variable_lower_bounds: Optional[np.ndarray] = None
    variable_upper_bounds: Optional[np.ndarray] = None
    constraint_lower_bounds: Optional[np.ndarray] = None
    constraint_upper_bounds: Optional[np.ndarray] = None
    variable_names: Optional[List[str]] = None
    constraint_names: Optional[List[str]] = None

    @property
    def nnz(self) -> int:
        return int(self.values.size) if self.values is not None else 0

    def copy(self) -> "LPProblem":
        return LPProblem(
            num_variables=self.num_variables,
            num_constraints=self.num_constraints,
            row_ptr=self.row_ptr.copy(),
            col_indices=self.col_indices.copy(),
            values=self.values.copy(),
            objective=self.objective.copy(),
            objective_constant=self.objective_constant,
            variable_lower_bounds=(None if self.variable_lower_bounds is None
                                    else self.variable_lower_bounds.copy()),
            variable_upper_bounds=(None if self.variable_upper_bounds is None
                                    else self.variable_upper_bounds.copy()),
            constraint_lower_bounds=(None if self.constraint_lower_bounds is None
                                     else self.constraint_lower_bounds.copy()),
            constraint_upper_bounds=(None if self.constraint_upper_bounds is None
                                     else self.constraint_upper_bounds.copy()),
            variable_names=(None if self.variable_names is None
                            else list(self.variable_names)),
            constraint_names=(None if self.constraint_names is None
                              else list(self.constraint_names)),
        )

    def scipy_csr(self):
        """Return scipy.sparse.csr_matrix view of A (scipy imported lazily)."""
        import scipy.sparse as sp
        return sp.csr_matrix((self.values, self.col_indices, self.row_ptr),
                             shape=(self.num_constraints, self.num_variables))


# ---------------------------------------------------------------------------
# Gurobi frontend: extract the LP part of an existing gurobi model.
# alpha-beta-CROWN already builds its LP/MIP relaxations in gurobipy
# (auto_LiRPA solver_module + lp_mip_solver). We reuse that model builder
# and only replace the solve engine, so no graph-to-LP code is duplicated.
# ---------------------------------------------------------------------------

def gurobi_to_mlxpdlp(model, objective_vars=None, objective_coeffs=None) -> LPProblem:
    """Convert a gurobipy model (LP part only) to an mlxPDLP LPProblem.

    Args:
        model: gurobipy.Model. Must be a pure LP (all continuous) - integer
            variables are NOT supported by mlxPDLP and are rejected.
        objective_vars: optional list of Var to use as objective instead of
            the model's current objective (per-neuron LP pattern: the code
            calls model.setObjective(v) before optimize; passing the vars
            here lets callers keep the model untouched).
        objective_coeffs: optional coeffs aligned with objective_vars.
    """
    try:
        import gurobipy as grb
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "gurobi_to_mlxpdlp requires gurobipy, which is not installed") from e

    vars_ = model.getVars()
    if any(v.VType != grb.GRB.CONTINUOUS for v in vars_):
        raise ValueError(
            "mlxPDLP solves LPs only; the gurobi model has non-continuous "
            "variables (a MIP). Refusing to convert.")

    num_variables = len(vars_)

    # getA() and getConstrs() only reflect constraints as of the last
    # update(); pending addConstr() calls would otherwise be silently
    # dropped from the LP. Update first, then read both.
    update = getattr(model, "update", None)
    if callable(update):
        update()
    num_constraints = len(model.getConstrs())
    # Constraint matrix (scipy CSR; rows follow model.getConstrs() order).
    A = model.getA()
    if A.shape[0] != num_constraints:
        try:
            model.write("/tmp/mlxpdlp_dbg_model.lp")
            with open("/tmp/mlxpdlp_dbg_names.txt", "w") as _f:
                _f.write("\n".join(c.ConstrName for c in model.getConstrs()))
        except Exception:
            pass
        raise RuntimeError(
            f"gurobi model matrix has {A.shape[0]} rows but "
            f"{num_constraints} constraints after update() "
            f"(NumConstrs={getattr(model, 'NumConstrs', '?')}, "
            f"NumGenConstrs={getattr(model, 'NumGenConstrs', '?')}, "
            f"NumQConstrs={getattr(model, 'NumQConstrs', '?')}); "
            "cannot convert safely")
    lp = LPProblem(
        num_variables=num_variables,
        num_constraints=num_constraints,
        row_ptr=np.asarray(A.indptr, dtype=np.int64),
        col_indices=np.asarray(A.indices, dtype=np.int64),
        values=np.asarray(A.data, dtype=np.float64),
        objective=np.zeros(num_variables, dtype=np.float64),
        objective_constant=0.0,
    )

    if objective_vars is None:
        # Use the model's current objective.
        objective_vars, objective_coeffs = [], []
        for i, v in enumerate(vars_):
            if v.Obj != 0.0:
                objective_vars.append(v)
                objective_coeffs.append(v.Obj)
        if model.ModelSense == grb.GRB.MAXIMIZE:
            objective_coeffs = [-c for c in objective_coeffs]
        objective_constant = model.ObjCon if model.ModelSense == grb.GRB.MINIMIZE else -model.ObjCon
    else:
        if objective_coeffs is None:
            objective_coeffs = [1.0] * len(objective_vars)
        assert len(objective_vars) == len(objective_coeffs)
        # objective_vars given explicitly are assumed in minimize sense.
        objective_constant = 0.0

    var_index = {v: i for i, v in enumerate(vars_)}
    for v, coeff in zip(objective_vars, objective_coeffs):
        lp.objective[var_index[v]] = coeff
    lp.objective_constant = float(objective_constant)

    inf = float("inf")
    lb = np.array([v.LB for v in vars_], dtype=np.float64)
    ub = np.array([v.UB for v in vars_], dtype=np.float64)
    # Gurobi uses +-GRB_INFINITY (~1e30) sentinels; normalize them.
    lb[lb < -1e20] = -inf
    ub[ub > 1e20] = inf
    lp.variable_lower_bounds = lb
    lp.variable_upper_bounds = ub

    constrs = model.getConstrs()
    rhs = np.array([c.RHS for c in constrs], dtype=np.float64)
    c_lb = np.full(num_constraints, -inf, dtype=np.float64)
    c_ub = np.full(num_constraints, inf, dtype=np.float64)
    for i, c in enumerate(constrs):
        if c.Sense == grb.GRB.EQUAL:
            c_lb[i] = c_ub[i] = rhs[i]
        elif c.Sense == grb.GRB.LESS_EQUAL:
            c_ub[i] = rhs[i]
        elif c.Sense == grb.GRB.GREATER_EQUAL:
            c_lb[i] = rhs[i]
        else:  # GRB.RANGE
            c_lb[i] = rhs[i]
            c_ub[i] = c.RHS + c.SARHSUp
    lp.constraint_lower_bounds = c_lb
    lp.constraint_upper_bounds = c_ub

    lp.variable_names = [v.VarName for v in vars_]
    lp.constraint_names = [c.ConstrName for c in constrs]
    return lp


# ---------------------------------------------------------------------------
# Solve wrapper
# ---------------------------------------------------------------------------

def make_parameters(tol: float = 1e-4, time_limit: float = 3600.0,
                    iteration_limit: int = 0, presolve: bool = True,
                    verbose: bool = False, ruiz_iterations: Optional[int] = None,
                    restart_policy: Optional[int] = None,
                    seed_defaults: bool = True):
    """Build mlxpdlp.Parameters with the requested optimality tolerance.

    ruiz_iterations/restart_policy: Phase-3 tuning knobs (on network LPs,
    ruiz_iterations=0 is ~2.2x faster than the 10 default).
    """
    import mlxpdlp
    params = mlxpdlp.Parameters()
    params.verbose = verbose
    params.presolve = presolve
    params.tolerance = tol
    params.termination_criteria.eps_optimal_relative = tol
    params.termination_criteria.eps_feasible_relative = tol
    params.time_limit_seconds = time_limit
    if ruiz_iterations is not None:
        params.l_inf_ruiz_iterations = ruiz_iterations
    if restart_policy is not None:
        params.restart_policy = restart_policy
    if iteration_limit > 0:
        params.iteration_limit = iteration_limit
    return params


def solve_mlxpdlp(lp: LPProblem, device: str = "cpu", tol: float = 1e-4,
                  time_limit: float = 3600.0, presolve: bool = True,
                  verbose: bool = False, warm_start: Optional[Dict] = None,
                  parameters=None, prescale: str = "auto",
                  ruiz_iterations: Optional[int] = None,
                  restart_policy: Optional[int] = None,
                  host_polish: bool = False) -> Dict:
    """Solve an LPProblem with mlxPDLP.

    Returns a dict with the raw result plus host-float64 audit fields:
      result          - mlxpdlp.SolveResult
      objective       - primal objective value
      dual_objective  - dual objective value
      status          - TerminationReason name
      time_sec        - cumulative solve time
      cert_lb         - certified lower bound for min objective, or None if
                        the certificate failed the host FP64 audit
      cert_lb_ok      - bool
      primal_viol     - max host FP64 primal violation (constraints + bounds)
      audit_gap       - relative stationarity residual ||c - A'y - z||
    """
    import mlxpdlp

    if device in ("metal",):
        device = "gpu"
    if device == "gpu" and not mlxpdlp.has_gpu():
        raise RuntimeError("mlxPDLP: Metal device requested but has_gpu() is False")

    if parameters is None:
        parameters = make_parameters(tol=tol, time_limit=time_limit,
                                     presolve=presolve, verbose=verbose,
                                     ruiz_iterations=ruiz_iterations,
                                     restart_policy=restart_policy)
    elif time_limit is not None:
        # A caller-provided Parameters object must not bypass the current
        # stage's remaining deadline.
        parameters.time_limit_seconds = max(0.0, float(time_limit))
    if host_polish and parameters is not None:
        # Opt-in 1e-5 Metal accuracy: bounded host-FP64 correction after
        # the FP32 Metal iterations (mlxPDLP >= 2026-08-23).
        parameters.host_double_polishing = True
        parameters.host_double_early_handoff = True
    if warm_start is not None and parameters.presolve:
        raise ValueError("mlxPDLP warm starts require parameters.presolve=False")

    # Conditioning guard (Phase 4): substitute x = lb + d x' for finite
    # two-sided variables when the bound-range ratio is extreme.
    mapping = None
    work_lp = lp
    if prescale in ("auto", "on"):
        prescaled = prescale_lp_variables(lp)
        if prescaled is not None:
            work_lp, mapping = prescaled
            if prescale == "auto":
                print(f"mlxPDLP: prescaled {work_lp.num_variables} variables "
                      "(bound-range ratio > 1e6)")

    # The public warm-start contract uses original-model coordinates. Apply
    # the inverse of the variable prescaling before constructing the solver;
    # row duals are unchanged, while reduced costs scale like the objective.
    solver_warm_start = warm_start
    if mapping is not None and warm_start is not None:
        solver_warm_start = map_warm_start_to_scaled(warm_start, mapping)

    solver = mlxpdlp.Solver(
        num_variables=work_lp.num_variables,
        num_constraints=work_lp.num_constraints,
        row_ptr=work_lp.row_ptr,
        col_indices=work_lp.col_indices,
        values=work_lp.values,
        variable_lower_bounds=work_lp.variable_lower_bounds,
        variable_upper_bounds=work_lp.variable_upper_bounds,
        constraint_lower_bounds=work_lp.constraint_lower_bounds,
        constraint_upper_bounds=work_lp.constraint_upper_bounds,
        objective=work_lp.objective,
        objective_constant=work_lp.objective_constant,
        parameters=parameters,
        primal_start=(solver_warm_start.get("primal")
                      if solver_warm_start else None),
        dual_start=(solver_warm_start.get("dual")
                    if solver_warm_start else None),
        reduced_cost_start=(solver_warm_start.get("reduced_cost")
                            if solver_warm_start else None),
        device=device,
    )
    res = solver.solve()

    cert_gate = max(1e-6, tol) if (host_polish and device == "gpu") else 1e-6
    audit = audit_certificate(lp, res, cert_gate=cert_gate)
    if mapping is not None:
        # Re-audit on the ORIGINAL model with the mapped certificate and
        # mapped primal. The returned `result` is replaced by a proxy in
        # ORIGINAL coordinates so downstream x-maps/warm starts are right.
        proxy = map_scaled_result_to_original(lp, res, mapping)
        audit = audit_certificate(lp, proxy, cert_gate=cert_gate)
        proxy.primal_objective_value = float(res.primal_objective_value)
        for attr in ("termination_reason", "termination_reason_name",
                     "total_count", "cumulative_time_sec",
                     "rescaling_time_sec", "presolve_time",
                     "relative_primal_residual", "relative_dual_residual",
                     "relative_objective_gap", "objective_gap",
                     "num_variables", "num_constraints", "num_nonzeros",
                     "num_reduced_variables", "num_reduced_constraints",
                     "num_reduced_nonzeros", "feasibility_polishing_time",
                     "feasibility_iteration",
                     "host_double_polishing_time",
                     "host_double_polishing_iteration",
                     "host_double_handoff"):
            if hasattr(res, attr):
                setattr(proxy, attr, getattr(res, attr))
        res = proxy
    return {
        "result": res,
        "objective": float(res.primal_objective_value),
        "dual_objective": float(res.dual_objective_value),
        "status": str(res.termination_reason_name),
        "time_sec": float(res.cumulative_time_sec),
        "num_iter": int(res.total_count),
        "primal_viol": audit["primal_viol"],
        "cert_lb": audit["cert_lb"],
        "cert_lb_ok": audit["cert_lb_ok"],
        "audit_gap": audit["audit_gap"],
    }


def audit_certificate(lp: LPProblem, result, cert_gate: float = 1e-6) -> Dict:
    """Re-audit a dual candidate on the original LP in host float64.

    The returned multipliers are first projected onto the sign constraints
    implied by finite row/variable bounds. The dual objective is then rebuilt
    from those projected multipliers; ``result.dual_objective_value`` is only
    diagnostic and is never trusted as a proof.

    A residual ``r = c - A.T @ y - z`` cannot be ignored merely because it is
    below a tolerance: if any variable is unbounded, ``r.T @ x`` can be
    arbitrarily large. Therefore a lower bound is returned only when either
    stationarity is exact in the host audit, or every variable has finite
    bounds and ``||r||_inf * max ||x||_1`` can be subtracted.
    """
    A = lp.scipy_csr()
    c = np.asarray(lp.objective, dtype=np.float64)
    con_lb = np.asarray(lp.constraint_lower_bounds, dtype=np.float64)
    con_ub = np.asarray(lp.constraint_upper_bounds, dtype=np.float64)
    var_lb = np.asarray(lp.variable_lower_bounds, dtype=np.float64)
    var_ub = np.asarray(lp.variable_upper_bounds, dtype=np.float64)
    y_raw = np.asarray(result.dual_solution, dtype=np.float64)
    z_raw = np.asarray(result.reduced_cost, dtype=np.float64)
    if y_raw.shape != (lp.num_constraints,) or z_raw.shape != (lp.num_variables,):
        raise ValueError("mlxPDLP dual certificate has incompatible dimensions")

    # mlxPDLP convention: positive multipliers select a lower bound and
    # negative multipliers select an upper bound. Projecting onto this cone
    # yields a dual-sign-feasible candidate even in the presence of tiny
    # numerical sign errors.
    y = y_raw.copy()
    y[~np.isfinite(con_lb)] = np.minimum(y[~np.isfinite(con_lb)], 0.0)
    y[~np.isfinite(con_ub)] = np.maximum(y[~np.isfinite(con_ub)], 0.0)
    z = z_raw.copy()
    z[~np.isfinite(var_lb)] = np.minimum(z[~np.isfinite(var_lb)], 0.0)
    z[~np.isfinite(var_ub)] = np.maximum(z[~np.isfinite(var_ub)], 0.0)
    sign_delta = np.concatenate((np.abs(y_raw - y), np.abs(z_raw - z)))
    dual_sign_viol = float(np.max(sign_delta)) if sign_delta.size else 0.0

    stationarity = c - A.T @ y - z
    norm_c = max(float(np.max(np.abs(c))) if c.size else 0.0, 1.0)
    stationarity_abs = (float(np.max(np.abs(stationarity)))
                        if stationarity.size else 0.0)
    audit_gap = stationarity_abs / norm_c

    x_raw = getattr(result, "primal_solution", None)
    if x_raw is None:
        primal_viol = float("inf")
    else:
        x = np.asarray(x_raw, dtype=np.float64)
        if x.shape != (lp.num_variables,) or not np.isfinite(x).all():
            primal_viol = float("inf")
        else:
            Ax = A @ x
            primal_viol = float(max(
                np.max(np.abs(np.clip(Ax, con_lb, con_ub) - Ax))
                if Ax.size else 0.0,
                np.max(np.abs(np.clip(x, var_lb, var_ub) - x))
                if x.size else 0.0,
            ))

    # Avoid 0 * infinity while rebuilding the bound-dual objective.
    dual_objective = float(lp.objective_constant)
    y_pos = y > 0.0
    y_neg = y < 0.0
    z_pos = z > 0.0
    z_neg = z < 0.0
    dual_objective += float(np.dot(con_lb[y_pos], y[y_pos]))
    dual_objective += float(np.dot(con_ub[y_neg], y[y_neg]))
    dual_objective += float(np.dot(var_lb[z_pos], z[z_pos]))
    dual_objective += float(np.dot(var_ub[z_neg], z[z_neg]))

    reported_dual = float(getattr(result, "dual_objective_value", np.nan))
    reported_gap = (abs(reported_dual - dual_objective)
                    if np.isfinite(reported_dual) else float("inf"))

    if np.isfinite(var_lb).all() and np.isfinite(var_ub).all():
        radius = float(np.sum(np.maximum(np.abs(var_lb), np.abs(var_ub))))
    else:
        radius = None

    raw_cert_lb = None
    if stationarity_abs == 0.0 and np.isfinite(dual_objective):
        raw_cert_lb = float(np.nextafter(dual_objective, -np.inf))
    safe_lb = None
    if radius is not None and np.isfinite(dual_objective):
        corrected = dual_objective - stationarity_abs * radius
        safe_lb = float(np.nextafter(corrected, -np.inf))

    # Prefer the finite-box correction because it remains sound for a
    # nonzero residual. cert_gate is retained as a convergence diagnostic,
    # never as an acceptance rule.
    cert_lb = safe_lb if safe_lb is not None else raw_cert_lb
    return {
        "audit_gap": audit_gap,
        "audit_tight": audit_gap <= cert_gate,
        "stationarity_abs": stationarity_abs,
        "dual_sign_viol": dual_sign_viol,
        "dual_objective": dual_objective,
        "reported_dual_objective_gap": reported_gap,
        "primal_radius": radius,
        "primal_viol": primal_viol,
        "raw_cert_lb": raw_cert_lb,
        "safe_lb": safe_lb,
        "cert_lb": cert_lb,
        "cert_lb_ok": cert_lb is not None,
    }


def primal_violation(lp: LPProblem, result) -> float:
    """Max host-float64 primal violation of a SolveResult (constraints + bounds)."""
    return float(audit_certificate(lp, result)["primal_viol"])


# ---------------------------------------------------------------------------
# Phase 1: certified wrapper with Gurobi-compatible statuses and a
# margin-based escalation ladder.
#
# Findings from Phase 0 that this encodes:
#   * Metal FP32 is the fast engine but its certificates are loose
#     (~1e-3..6e-3 relative after the rigorous correction); CPU FP64 at
#     tol 1e-5 is the certified engine (essentially exact certificates,
#     competitive with HiGHS at medium+, solves large where HiGHS gives
#     nothing in 120s).
#   * tol 1e-7 on CPU is a trap (300s TIME_LIMIT while already converged).
#   * Warm starts do not pay off when per-neuron objectives change. The
#     same-objective Metal -> CPU fallback does reuse Metal's final iterate.
#   * Primal solutions from Metal violate constraints by ~1e-4..5e-3
#     (absolute 0.01..0.4 at large) - counterexamples need a feasibility
#     check before they may declare "unsafe".
#
# Soundness contract for the alpha-beta-CROWN call sites:
#   * Solver termination, an audited dual bound, and a feasible primal are
#     independent facts. They are exposed as status, bound_ok/bound, and
#     solcount/objective/x respectively; call sites must use the right fact.
#   * A nonzero dual residual is only accepted after the finite-box correction.
#   * INFEASIBLE(3) is only reported from an mlxPDLP primal-infeasibility
#     certificate.
#   * TIME_LIMIT(9) or UNKNOWN(-1) may still carry a sound bound or feasible
#     primal. Neither fact changes the termination status, and neither may be
#     mistaken for the other (especially in zero-objective feasibility calls).
# ---------------------------------------------------------------------------

# Gurobi-compatible status codes (same integers as gurobipy GRB.Status).
STATUS_OPTIMAL = 2
STATUS_INFEASIBLE = 3
STATUS_INF_OR_UNBD = 4
STATUS_UNBOUNDED = 5
STATUS_TIME_LIMIT = 9
STATUS_INTERRUPTED = 11
STATUS_USER_OBJ_LIMIT = 15
STATUS_UNKNOWN = -1


def mlxpdlp_status_to_gurobi(termination_reason, name: str = "") -> int:
    """Map an mlxPDLP TerminationReason (int/enum or name string) to a
    Gurobi-compatible status integer."""
    if isinstance(termination_reason, str):
        name = termination_reason
        termination_reason = None
    else:
        try:
            import mlxpdlp
            if name == "":
                name = str(mlxpdlp.TerminationReason(termination_reason).name)
        except (ImportError, ValueError):
            pass
    name = (name or "").upper()
    if termination_reason is not None:
        try:
            import mlxpdlp
            reason = mlxpdlp.TerminationReason
            if termination_reason == reason.OPTIMAL or name == "OPTIMAL":
                return STATUS_OPTIMAL
            if termination_reason == reason.PRIMAL_INFEASIBLE or name == "PRIMAL_INFEASIBLE":
                return STATUS_INFEASIBLE
            if termination_reason == reason.DUAL_INFEASIBLE or name == "DUAL_INFEASIBLE":
                return STATUS_UNBOUNDED
            if termination_reason == reason.INFEASIBLE_OR_UNBOUNDED or name == "INFEASIBLE_OR_UNBOUNDED":
                return STATUS_INF_OR_UNBD
            if termination_reason in (reason.TIME_LIMIT, reason.ITERATION_LIMIT)                     or name in ("TIME_LIMIT", "ITERATION_LIMIT"):
                return STATUS_TIME_LIMIT
            if name == "FEAS_POLISH_SUCCESS":
                return STATUS_OPTIMAL
            if name == "HOST_DOUBLE_HANDOFF":
                return STATUS_TIME_LIMIT
        except Exception:
            pass
    mapping = {
        "OPTIMAL": STATUS_OPTIMAL,
        "PRIMAL_INFEASIBLE": STATUS_INFEASIBLE,
        "DUAL_INFEASIBLE": STATUS_UNBOUNDED,
        "INFEASIBLE_OR_UNBOUNDED": STATUS_INF_OR_UNBD,
        "TIME_LIMIT": STATUS_TIME_LIMIT,
        "ITERATION_LIMIT": STATUS_TIME_LIMIT,
        "INTERRUPTED": STATUS_INTERRUPTED,
        "FEAS_POLISH_SUCCESS": STATUS_OPTIMAL,
        "HOST_DOUBLE_HANDOFF": STATUS_TIME_LIMIT,
    }
    return mapping.get(name, STATUS_UNKNOWN)


@dataclass
class CertifiedSolve:
    """A solve result in Gurobi-compatible form with a certified bound."""
    status: int
    objective: Optional[float]        # sense-corrected primal objective
    bound: Optional[float]            # certified bound: lower for min, upper for max
    bound_ok: bool                    # True when ``bound`` passed the host audit
    audit_gap: float                  # relative stationarity of the certificate
    stationarity_abs: float
    primal_viol: float                # host-FP64 max primal violation
    time_sec: float
    num_iter: int
    device: str
    termination: str = ""
    solcount: int = 0                 # 1 when a feasible primal exists
    x: Optional[np.ndarray] = None    # primal solution (original model coords)
    raw: Dict = field(default_factory=dict)

    @property
    def optimal(self) -> bool:
        return self.status == STATUS_OPTIMAL

    @property
    def infeasible(self) -> bool:
        return self.status == STATUS_INFEASIBLE


def _extract_metal_warm_start(csolve: CertifiedSolve,
                              lp: LPProblem) -> Optional[Dict]:
    """Return a validated original-coordinate warm start from a Metal solve.

    ``_solve_one`` normalizes both min/max objectives before calling mlxPDLP,
    so these primal/dual iterates already correspond to the exact objective
    the CPU fallback will solve. Results from a prescaled solve are mapped
    back to original coordinates by ``solve_mlxpdlp`` first.
    """
    if csolve.device not in ("gpu", "metal"):
        return None
    result = csolve.raw.get("result") if csolve.raw else None
    if result is None:
        return None
    values = (
        getattr(result, "primal_solution", None),
        getattr(result, "dual_solution", None),
    )
    if any(value is None for value in values):
        return None
    try:
        primal, dual = (
            np.asarray(value, dtype=np.float64).reshape(-1).copy()
            for value in values
        )
    except (TypeError, ValueError):
        return None
    if (primal.size != lp.num_variables or
            dual.size != lp.num_constraints):
        return None
    if not (np.all(np.isfinite(primal)) and np.all(np.isfinite(dual))):
        return None
    return {
        "primal": primal,
        "dual": dual,
    }


def _solve_one(lp: LPProblem, device: str, tol: float, time_limit: float,
               sense: str, warm_start=None, parameters=None,
               iteration_limit: int = 0, prescale: str = "auto",
               ruiz_iterations: Optional[int] = None,
               restart_policy: Optional[int] = None,
               host_polish: bool = False) -> CertifiedSolve:
    """Single-backend solve + audit; objective negated for sense='max'."""
    lp_work = lp
    if sense == "max":
        lp_work = lp.copy()
        lp_work.objective = -lp_work.objective
        lp_work.objective_constant = -lp_work.objective_constant
    out = solve_mlxpdlp(lp_work, device=device, tol=tol, time_limit=time_limit,
                        warm_start=warm_start, parameters=parameters,
                        prescale=prescale, ruiz_iterations=ruiz_iterations,
                        restart_policy=restart_policy, host_polish=host_polish)
    res = out["result"]
    # Re-audit on the exact objective/sense used for this stage.
    cert_gate = max(1e-6, tol) if (host_polish and device == "gpu") else 1e-6
    audit = audit_certificate(lp_work, res, cert_gate=cert_gate)
    status = mlxpdlp_status_to_gurobi(res.termination_reason,
                                      str(res.termination_reason_name))
    bound = audit["cert_lb"]
    bound_ok = bound is not None
    # Preserve the solver's termination semantics. In particular, an
    # UNKNOWN termination with a useful dual bound is still UNKNOWN; a bound
    # is not evidence that a primal feasible point (or optimum) exists.
    if status == STATUS_OPTIMAL and not bound_ok:
        status = STATUS_TIME_LIMIT  # not decidable at this tolerance
    objective = out["objective"]
    if sense == "max":
        objective = -objective
        bound = None if bound is None else -bound  # upper bound
    return CertifiedSolve(
        status=status,
        objective=float(objective) if objective is not None else None,
        bound=float(bound) if bound is not None else None,
        bound_ok=bound_ok,
        audit_gap=audit["audit_gap"],
        stationarity_abs=audit["stationarity_abs"],
        primal_viol=audit["primal_viol"],
        solcount=1 if audit["primal_viol"] <= 1e-4 else 0,
        time_sec=out["time_sec"],
        num_iter=out["num_iter"],
        device=device,
        termination=str(res.termination_reason_name),
        x=(np.asarray(res.primal_solution, dtype=np.float64)
           if res.primal_solution is not None else None),
        raw=out,
    )


def solve_gurobi_lp(lp: LPProblem, time_limit: float = 3600.0,
                    sense: str = "min") -> CertifiedSolve:
    """Exact fallback: rebuild the LP in gurobipy and solve it.

    Requires a Gurobi license. Returns the Gurobi optimum as a
    CertifiedSolve with an exactly-certified bound (simplex/IPM optimum).
    """
    try:
        import gurobipy as grb
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("gurobi fallback requires gurobipy") from e

    import time as _time
    t0 = _time.time()
    model = grb.Model("mlxpdlp_fallback")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    A = lp.scipy_csr()
    vars_ = model.addMVar(
        lp.num_variables,
        lb=lp.variable_lower_bounds, ub=lp.variable_upper_bounds,
        name="x")
    senses = []
    rhs = []
    clb = lp.constraint_lower_bounds
    cub = lp.constraint_upper_bounds
    for i in range(lp.num_constraints):
        if clb[i] == cub[i]:
            senses.append("=")
            rhs.append(cub[i])
        elif cub[i] == float("inf"):
            senses.append(">")
            rhs.append(clb[i])
        elif clb[i] == -float("inf"):
            senses.append("<")
            rhs.append(cub[i])
        else:
            senses.append("<")
            rhs.append(cub[i])
            model.addConstr(A.getrow(i) @ vars_ >= clb[i])
    model.addMConstr(A, vars_, senses, np.asarray(rhs, dtype=np.float64))
    obj = lp.objective if sense == "min" else -lp.objective
    objective_constant = (lp.objective_constant if sense == "min"
                          else -lp.objective_constant)
    model.setMObjective(None, obj, objective_constant,
                        sense=grb.GRB.MINIMIZE)
    model.optimize()
    status = int(model.Status)
    objective = None
    bound = None
    primal_viol = 0.0
    x_sol = None
    solcount = int(getattr(model, "SolCount", status == STATUS_OPTIMAL))
    if solcount:
        normalized_objective = float(model.ObjVal)
        objective = (normalized_objective if sense == "min"
                     else -normalized_objective)
        try:
            x_sol = np.asarray(vars_.X, dtype=np.float64)
        except Exception:
            x_sol = None
    # ObjBound is the solver's dual bound. ObjVal is a primal incumbent and
    # must never be substituted for it, especially on an early termination.
    if status not in (STATUS_INFEASIBLE, STATUS_INF_OR_UNBD,
                      STATUS_UNBOUNDED):
        try:
            normalized_bound = float(model.ObjBound)
            if np.isfinite(normalized_bound):
                bound = (normalized_bound if sense == "min"
                         else -normalized_bound)
        except Exception:
            bound = None
    elif status == STATUS_INFEASIBLE:
        bound = float("inf") if sense == "min" else -float("inf")
    return CertifiedSolve(
        status=status,
        objective=objective,
        bound=bound,
        bound_ok=bound is not None,
        audit_gap=0.0,
        stationarity_abs=0.0,
        primal_viol=primal_viol,
        solcount=1 if solcount and x_sol is not None else 0,
        time_sec=_time.time() - t0,
        num_iter=0,
        device="gurobi",
        termination="GUROBI",
        x=x_sol,
    )


def solve_certified(lp: LPProblem, device: str = "gpu", tol: float = 1e-4,
                    time_limit: float = 3600.0, sense: str = "min",
                    margin: Optional[float] = None,
                    threshold: Optional[float] = None,
                    warm_start: Optional[Dict] = None,
                    fallback=None, parameters=None, prescale: str = "auto",
                    ruiz_iterations: Optional[int] = None,
                    restart_policy: Optional[int] = None,
                    host_polish: bool = False) -> CertifiedSolve:
    """Certified LP solve with a margin-based escalation ladder.

    Args:
        lp: LPProblem (minimize convention internally).
        device: "gpu" (Metal FP32) or "cpu" (FP64).
        tol: optimality tolerance (1e-4 Metal default, 1e-5 CPU recommended).
        time_limit: total budget for the initial solve and every fallback.
        sense: "min" or "max" (objective negated internally for max).
        margin, threshold: when both given, the solve is only considered
            decisive if the certified bound proves
              min: bound >= threshold + margin
              max: bound <= threshold - margin
            (the bound is on the safe side, so this is sound). Otherwise
            the escalation ladder runs.
        warm_start: dict with primal/dual/reduced_cost starts (CPU only;
            requires presolve=False parameters).
        fallback: None -> default ladder ["cpu" @ 1e-6, then gurobi];
            a list of callables lp -> CertifiedSolve; a single callable;
            or the strings "cpu"/"gurobi" (default ladder for that entry).
        parameters: mlxpdlp.Parameters override for the FIRST stage.

    Returns CertifiedSolve. Callers must check .status and .bound_ok:
      STATUS_OPTIMAL(2)   -> the solver reached an audited optimum
      STATUS_INFEASIBLE(3)-> proven infeasible
      otherwise           -> not decided, though .bound may still prove a
                             threshold decision
    """
    assert sense in ("min", "max"), sense
    if time_limit is not None and float(time_limit) <= 0.0:
        return CertifiedSolve(
            status=STATUS_TIME_LIMIT,
            objective=None,
            bound=None,
            bound_ok=False,
            audit_gap=float("inf"),
            stationarity_abs=float("inf"),
            primal_viol=float("inf"),
            time_sec=0.0,
            num_iter=0,
            device=device,
            termination="TIME_LIMIT",
        )
    decisive_margin = margin is not None and threshold is not None

    import time as _time
    started = _time.monotonic()

    def remaining_time():
        if time_limit is None:
            return None
        return max(0.0, float(time_limit) - (_time.monotonic() - started))

    def ladder():
        if fallback is None:
            yield ("cpu", 1e-6)
            yield ("gurobi", None)
        elif callable(fallback):
            yield fallback
        else:
            for fb in (fallback if isinstance(fallback, (list, tuple)) else [fallback]):
                if callable(fb):
                    yield fb
                elif fb == "cpu":
                    yield ("cpu", 1e-6)
                elif fb == "gurobi":
                    yield ("gurobi", None)
                elif fb == "highs":
                    yield ("highs", None)
                else:
                    raise ValueError(f"unknown fallback spec {fb!r}")

    best = _solve_one(lp, device, tol, time_limit, sense,
                      warm_start=warm_start, parameters=parameters,
                      prescale=prescale, ruiz_iterations=ruiz_iterations,
                      restart_policy=restart_policy, host_polish=host_polish)
    results = [best]

    def decisive(cs: CertifiedSolve) -> bool:
        if not decisive_margin:
            return cs.status == STATUS_INFEASIBLE or (
                cs.status == STATUS_OPTIMAL and cs.bound_ok
                and cs.bound is not None)
        if cs.status == STATUS_INFEASIBLE:
            return True
        if not cs.bound_ok or cs.bound is None:
            return False
        if sense == "min":
            return cs.bound >= threshold + margin
        return cs.bound <= threshold - margin

    if decisive(best):
        return best

    # Copy Metal's final iterate only when escalation is actually required.
    # A same-objective CPU fallback can continue from it; other fallback
    # engines keep their existing cold-start behavior.
    metal_warm_start = _extract_metal_warm_start(best, lp)

    for fb in ladder():
        remaining = remaining_time()
        if remaining is not None and remaining <= 0.0:
            break
        try:
            if isinstance(fb, tuple) and fb[0] == "cpu":
                _, fb_tol = fb
                params = make_parameters(tol=fb_tol, time_limit=remaining,
                                         presolve=metal_warm_start is None,
                                         ruiz_iterations=ruiz_iterations,
                                         restart_policy=restart_policy)
                try:
                    cs = _solve_one(lp, "cpu", fb_tol, remaining, sense,
                                    warm_start=metal_warm_start,
                                    parameters=params, prescale=prescale)
                    if metal_warm_start is not None:
                        cs.raw["warm_started_from"] = "gpu"
                except Exception:
                    # A transferred iterate must never disable the existing
                    # certified CPU fallback. Retry cold when validation in
                    # the native solver rejects a seemingly usable start.
                    if metal_warm_start is None:
                        raise
                    cold_remaining = remaining_time()
                    if cold_remaining is not None and cold_remaining <= 0.0:
                        raise
                    cold_params = make_parameters(
                        tol=fb_tol, time_limit=cold_remaining, presolve=True,
                        ruiz_iterations=ruiz_iterations,
                        restart_policy=restart_policy)
                    cs = _solve_one(
                        lp, "cpu", fb_tol, cold_remaining, sense,
                        warm_start=None, parameters=cold_params,
                        prescale=prescale)
                    cs.raw["warm_start_retry"] = "cold"
            elif isinstance(fb, tuple) and fb[0] == "gurobi":
                cs = solve_gurobi_lp(lp, time_limit=remaining, sense=sense)
            elif isinstance(fb, tuple) and fb[0] == "highs":
                cs = solve_highs_lp(lp, time_limit=remaining, sense=sense)
            elif callable(fb):
                cs = fb(lp)
            else:
                continue
            results.append(cs)
            if decisive(cs):
                return cs
        except Exception:  # keep the ladder going; record nothing
            results.append(None)
            continue
    # Not decidable: return the most informative result. Later escalation
    # stages win ties (a CPU/gurobi result is at least as trustworthy as
    # the Metal fast path it escalated from).
    def _key(r):
        return (r.status == STATUS_INFEASIBLE,
                bool(r.bound_ok and r.bound is not None),
                r.status == STATUS_OPTIMAL, bool(r.solcount))
    best = None
    for r in results:
        if r is not None and (best is None or _key(r) >= _key(best)):
            best = r
    return best


def decide_safe(csolve: CertifiedSolve, threshold: float,
                margin: float = 0.0, sense: str = "min",
                counterexample_tol: float = 1e-5) -> str:
    """Sound verdict helper for LP-decision call sites.

    Returns:
      "safe"   - certified bound proves the threshold with margin
      "unsafe" - primal solution is feasible (viol <= counterexample_tol)
                 and its objective violates the threshold (only a hint of
                 a counterexample; verify it on the exact model)
      "unknown" - escalate / treat as inconclusive
    """
    if csolve.status == STATUS_INFEASIBLE:
        return "safe"
    if csolve.status == STATUS_OPTIMAL and csolve.bound is not None:
        if sense == "min":
            return "safe" if csolve.bound >= threshold + margin else "unknown"
        return "safe" if csolve.bound <= threshold - margin else "unknown"
    # Non-optimal statuses can still certify safety through the bound.
    if csolve.bound is not None:
        if sense == "min" and csolve.bound >= threshold + margin:
            return "safe"
        if sense == "max" and csolve.bound <= threshold - margin:
            return "safe"
    if csolve.objective is not None and csolve.primal_viol <= counterexample_tol:
        if sense == "min" and csolve.objective <= threshold - margin:
            return "unsafe"
        if sense == "max" and csolve.objective >= threshold + margin:
            return "unsafe"
    return "unknown"


def counterexample_feasible(lp: LPProblem, result, tol: float = 1e-5) -> bool:
    """True when a raw solve's primal solution is feasible to `tol` on the
    original model (required before using it as an "unsafe" witness)."""
    return primal_violation(lp, result) <= tol


# ---------------------------------------------------------------------------
# Phase 2: Gurobi-frontend integration API.
#
# alpha-beta-CROWN keeps building its LP models in gurobipy (solver_module +
# lp_mip_solver). These helpers let the LP call sites swap ONLY the solve
# engine: the current gurobi model state (bounds, added rows, objective) is
# converted to an LPProblem and solved through the Phase-1 certified
# wrapper. Call sites stay unchanged for the default gurobi backend.
# ---------------------------------------------------------------------------

@dataclass
class GurobiLikeResult:
    """Emulates the attributes alpha-beta-CROWN reads off a solved gurobi
    model (model.status, var.X, model.objval, model.objbound, solcount)."""
    status: int                     # Gurobi-compatible status code
    objval: Optional[float]         # sense-corrected FEASIBLE primal
                                    # objective (Gurobi incumbent semantics:
                                    # an upper bound on the min / lower
                                    # bound on the max); None if no
                                    # feasible primal exists
    objbound: float                 # certified bound (<= true min for min
                                    # sense, >= true max for max sense)
    solcount: int                   # 1 when a feasible primal exists
    x: Dict[str, float]             # variable name -> feasible primal value
    solve_time: float
    device: str
    certified: bool
    termination: str = ""
    audit_gap: float = 0.0           # relative stationarity of the certificate
    primal_viol: float = 0.0


def effective_model_time_limit(model, requested: float) -> float:
    """Cap a backend request by the Gurobi model's current TimeLimit.

    Per-neuron/refinement models often carry a much smaller limit than the
    global BaB timeout. Reading it here also covers call sites that predate the
    backend and only communicate their deadline through the model parameter.
    """
    limits = [float(requested)]
    try:
        value = model.Params.TimeLimit
        if isinstance(value, (int, float, np.integer, np.floating)):
            current = float(value)
            if np.isfinite(current) and current >= 0.0:
                limits.append(current)
    except Exception:
        try:
            info = model.getParamInfo("TimeLimit")
            value = info[2]
            if isinstance(value, (int, float, np.integer, np.floating)):
                current = float(value)
                if np.isfinite(current) and current >= 0.0:
                    limits.append(current)
        except Exception:
            pass
    return max(0.0, min(limits))


def solve_gurobi_model_with_mlxpdlp(model, objective_var=None, sense="min",
                                    device="gpu", tol=1e-4, time_limit=3600.0,
                                    margin=None, threshold=None,
                                    warm_start=None,
                                    fallback=None, prescale: str = "auto",
                                    ruiz_iterations: Optional[int] = None,
                                    restart_policy: Optional[int] = None,
                                    host_polish: bool = False) -> GurobiLikeResult:
    """Solve the CURRENT state of a gurobipy LP model with mlxPDLP.

    Args:
        model: gurobipy.Model (LP part only; all variables continuous).
        objective_var: gurobipy Var to minimize/maximize (per-neuron
            pattern). If None, the model's own objective/sense is used
            (gurobi_to_mlxpdlp normalizes MAXIMIZE to minimization).
        sense: "min" (default) or "max" (only used with objective_var).
        device/tol/time_limit/margin/threshold: see solve_certified. The
            default ladder (Metal -> CPU@1e-5 -> Gurobi) applies.
        warm_start: optional dict, see solve_certified.

    Returns GurobiLikeResult. Contracts the call sites rely on:
        * certified/objbound describe the audited dual bound independently of
          termination status.
        * objval and every x entry always describe the feasible primal, never
          a dual bound.
        * status == 3 means a primal-infeasibility certificate.
        * solcount == 1 only when the returned primal is feasible to
          ~1e-4 on the original model (usable for counterexample checks).
    """
    time_limit = effective_model_time_limit(model, time_limit)
    model_was_max = (objective_var is None
                     and getattr(model, "ModelSense", 1) == -1)
    if time_limit <= 0.0:
        result_sense = ("max" if model_was_max else
                        ("min" if objective_var is None else sense))
        return GurobiLikeResult(
            status=STATUS_TIME_LIMIT,
            objval=None,
            objbound=(-float("inf") if result_sense == "min"
                      else float("inf")),
            solcount=0,
            x={},
            solve_time=0.0,
            device=device,
            certified=False,
            termination="TIME_LIMIT",
            audit_gap=float("inf"),
            primal_viol=float("inf"),
        )
    if objective_var is not None:
        lp = gurobi_to_mlxpdlp(model, objective_vars=[objective_var],
                               objective_coeffs=[1.0])
        cs = solve_certified(lp, device=device, tol=tol, time_limit=time_limit,
                             sense=sense, margin=margin, threshold=threshold,
                             warm_start=warm_start, fallback=fallback,
                             prescale=prescale,
                             ruiz_iterations=ruiz_iterations,
                             restart_policy=restart_policy,
                             host_polish=host_polish)
    else:
        lp = gurobi_to_mlxpdlp(model)
        # model objective already normalized (MAXIMIZE flipped) by the
        # conversion; solve as a minimization.
        cs = solve_certified(lp, device=device, tol=tol, time_limit=time_limit,
                             sense="min", margin=margin, threshold=threshold,
                             warm_start=warm_start, fallback=fallback,
                             prescale=prescale,
                             ruiz_iterations=ruiz_iterations,
                             restart_policy=restart_policy,
                             host_polish=host_polish)

    primal = cs.x
    if primal is None and cs.raw.get("result") is not None:
        ps = cs.raw["result"].primal_solution
        if ps is not None:
            primal = np.asarray(ps, dtype=np.float64)

    x = {}
    if primal is not None and primal.size == lp.num_variables \
            and lp.variable_names is not None:
        x = dict(zip(lp.variable_names, primal.tolist()))

    solcount = 1 if (primal is not None and cs.primal_viol <= 1e-4) else 0
    raw_obj = None
    if primal is not None and solcount and lp.objective is not None \
            and primal.size == lp.num_variables:
        raw_obj = float(np.dot(lp.objective, primal)) + lp.objective_constant
    # Explicit objective variables are not normalized by the converter, so
    # their primal value already has the original sign even for sense=max.
    # A model-owned MAXIMIZE objective, by contrast, was negated during
    # conversion and must be mapped back here.
    if model_was_max:
        raw_obj = None if raw_obj is None else -raw_obj
    result_sense = ("max" if model_was_max else
                    ("min" if objective_var is None else sense))
    result_bound = cs.bound
    if model_was_max and result_bound is not None:
        result_bound = -result_bound
    return GurobiLikeResult(
        status=cs.status,
        objval=raw_obj,
        objbound=float(result_bound) if result_bound is not None
                 else (-float("inf") if result_sense == "min"
                       else float("inf")),
        solcount=solcount,
        x=x,
        solve_time=cs.time_sec,
        device=cs.device,
        certified=bool(cs.bound_ok and result_bound is not None),
        termination=cs.termination,
        audit_gap=cs.audit_gap,
        primal_viol=cs.primal_viol,
    )


def get_lp_backend_settings():
    """Read the mlxPDLP backend settings from arguments.Config when the
    verifier configuration is available; otherwise return safe defaults
    (gurobi backend = current behavior)."""
    settings = {
        "backend": "gurobi",
        "device": "gpu",
        "tol": 1e-4,
        "margin": 1e-3,
        "fallback": "gurobi",
        "time_limit": 3600.0,
        "ruiz_iterations": 0,
        "restart_policy": 0,
        "metal_polish": "auto",
    }
    try:
        import arguments
        mip_cfg = arguments.Config["solver"]["mip"]
        settings["backend"] = mip_cfg.get("lp_backend", "gurobi")
        settings["device"] = mip_cfg.get("mlxpdlp_device", "gpu")
        settings["tol"] = mip_cfg.get("mlxpdlp_tolerance", 1e-4)
        settings["margin"] = mip_cfg.get("mlxpdlp_margin", 1e-3)
        settings["fallback"] = mip_cfg.get("mlxpdlp_fallback", "gurobi")
        settings["ruiz_iterations"] = mip_cfg.get("mlxpdlp_ruiz", 0)
        settings["restart_policy"] = mip_cfg.get("mlxpdlp_restart_policy", 0)
        settings["metal_polish"] = mip_cfg.get("mlxpdlp_polish", "auto")
        timeout = arguments.Config["bab"].get("timeout", 3600.0)
        settings["time_limit"] = float(timeout) if timeout else 3600.0
    except Exception:
        pass
    # Normalize device names.
    if settings["device"] in ("metal", "gpu", "mps"):
        settings["device"] = "gpu"
    else:
        settings["device"] = "cpu"
    if settings["fallback"] == "none":
        settings["fallback"] = []
    elif settings["fallback"] == "cpu":
        settings["fallback"] = ["cpu"]
    elif settings["fallback"] == "highs":
        settings["fallback"] = ["highs"]
    else:
        settings["fallback"] = ["gurobi"] if settings["fallback"] == "gurobi" else None
    return settings


def mlxpdlp_enabled():
    """True when the verifier config selects the mlxPDLP LP backend."""
    return get_lp_backend_settings()["backend"] == "mlxpdlp"


def metal_polish_enabled(settings: Optional[Dict] = None) -> bool:
    """Whether the opt-in 1e-5 Metal accuracy path (bounded host-FP64
    correction) applies: settings["metal_polish"] == "on", or "auto" with
    a Metal device and tolerance <= 1e-5 (mlxPDLP >= 2026-08-23 reaches an
    audited 1e-5 on well-conditioned LPs; the portable Metal guarantee
    stays 1e-4)."""
    s = settings if settings is not None else get_lp_backend_settings()
    if s.get("metal_polish") == "on":
        return True
    if s.get("metal_polish") == "off":
        return False
    # auto: polish the Metal stage when a tighter-than-1e-4 tolerance is
    # requested (the FP32 floor without the FP64 correction).
    return s.get("device") == "gpu" and s.get("tol", 1e-4) <= 1e-5


def make_fallback_from_settings(settings):
    """Build the fallback argument for solve_certified from settings."""
    fb = settings.get("fallback")
    if fb is None:
        return None  # default ladder (cpu@1e-6 then gurobi)
    return fb


# ---------------------------------------------------------------------------
# Phase 3 prep: HiGHS fallback / reference (license-free, scipy).
#
# The restricted gurobipy license caps models at ~2000 vars/cons, so
# larger LP relaxations need a license-free reference solver: scipy HiGHS
# (dual simplex/IPM). It slots into the escalation ladder exactly like the
# gurobi fallback.
# ---------------------------------------------------------------------------

def _audit_highs_dual(lp, result, objective, objective_constant,
                      A_eq, b_eq, A_ub, b_ub):
    """Build a corrected lower bound from SciPy/HiGHS marginals.

    SciPy reports objective sensitivities (marginals): equality marginals are
    free, <=-row marginals are nonpositive, lower-bound marginals are
    nonnegative, and upper-bound marginals are nonpositive. With that
    convention stationarity is ``c - A.T @ marginal - bound_marginals = 0``
    and the dual objective is the dot product of each marginal with its RHS.
    """
    c = np.asarray(objective, dtype=np.float64)
    lb = np.asarray(lp.variable_lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.variable_upper_bounds, dtype=np.float64)

    eq_marg = np.asarray(result.eqlin.marginals, dtype=np.float64)
    ub_marg = np.minimum(
        np.asarray(result.ineqlin.marginals, dtype=np.float64), 0.0)
    lower_raw = np.asarray(result.lower.marginals, dtype=np.float64)
    upper_raw = np.asarray(result.upper.marginals, dtype=np.float64)
    lower_marg = np.where(np.isfinite(lb), np.maximum(lower_raw, 0.0), 0.0)
    upper_marg = np.where(np.isfinite(ub), np.minimum(upper_raw, 0.0), 0.0)

    stationarity = c.copy()
    dual_objective = float(objective_constant)
    if A_eq is not None:
        stationarity -= np.asarray(A_eq.T @ eq_marg).ravel()
        dual_objective += float(np.dot(np.asarray(b_eq), eq_marg))
    if A_ub is not None:
        stationarity -= np.asarray(A_ub.T @ ub_marg).ravel()
        dual_objective += float(np.dot(np.asarray(b_ub), ub_marg))
    stationarity -= lower_marg + upper_marg
    dual_objective += float(np.dot(lb[np.isfinite(lb)],
                                   lower_marg[np.isfinite(lb)]))
    dual_objective += float(np.dot(ub[np.isfinite(ub)],
                                   upper_marg[np.isfinite(ub)]))

    stationarity_abs = (float(np.max(np.abs(stationarity)))
                        if stationarity.size else 0.0)
    norm_c = max(float(np.max(np.abs(c))) if c.size else 0.0, 1.0)
    if np.isfinite(lb).all() and np.isfinite(ub).all():
        radius = float(np.sum(np.maximum(np.abs(lb), np.abs(ub))))
        bound = dual_objective - stationarity_abs * radius
    elif stationarity_abs == 0.0:
        bound = dual_objective
    else:
        bound = None
    if bound is not None and np.isfinite(bound):
        bound = float(np.nextafter(bound, -np.inf))
    else:
        bound = None
    return bound, stationarity_abs / norm_c, stationarity_abs


def solve_highs_lp(lp: LPProblem, time_limit: float = 3600.0,
                   sense: str = "min") -> CertifiedSolve:
    """Exact reference solve via scipy.optimize.linprog (HiGHS).

    HiGHS status mapping: 0 optimal -> 2; 2 infeasible -> 3; 3 unbounded
    -> 5; 1/4 limits -> 9. Bounds come from audited HiGHS dual marginals;
    the primal objective is never substituted for a proof.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("highs fallback requires scipy") from e

    import time as _time
    t0 = _time.time()
    A = lp.scipy_csr()
    clb = lp.constraint_lower_bounds
    cub = lp.constraint_upper_bounds

    eq = clb == cub
    up_only = (clb == -np.inf) & (cub != np.inf)
    lo_only = (clb != -np.inf) & (cub == np.inf)
    free = (clb == -np.inf) & (cub == np.inf)
    two_sided = ~(eq | up_only | lo_only | free)

    def vstack(base, block):
        import scipy.sparse as sp
        if base is None:
            return block.copy()
        return sp.vstack([base, block], format="csr")

    A_eq, b_eq, A_ub, b_ub = None, None, None, None
    if eq.any():
        A_eq = A[eq]
        b_eq = cub[eq]
    if up_only.any():
        A_ub = A[up_only]
        b_ub = cub[up_only]
    if lo_only.any():
        A_ub = vstack(A_ub, -A[lo_only])
        b_ub = np.concatenate([b_ub, -clb[lo_only]]) if b_ub is not None else -clb[lo_only]
    if two_sided.any():
        A_ub = vstack(A_ub, A[two_sided])
        b_ub = np.concatenate([b_ub, cub[two_sided]]) if b_ub is not None else cub[two_sided]
        A_ub = vstack(A_ub, -A[two_sided])
        b_ub = np.concatenate([b_ub, -clb[two_sided]]) if b_ub is not None else -clb[two_sided]

    obj = lp.objective if sense == "min" else -lp.objective
    objective_constant = (lp.objective_constant if sense == "min"
                          else -lp.objective_constant)
    options = ({"time_limit": max(0.0, float(time_limit))}
               if time_limit is not None else None)
    res = linprog(
        obj,
        A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        bounds=list(zip(lp.variable_lower_bounds, lp.variable_upper_bounds)),
        method="highs", options=options)

    status_map = {0: STATUS_OPTIMAL, 2: STATUS_INFEASIBLE, 3: STATUS_UNBOUNDED,
                  1: STATUS_TIME_LIMIT, 4: STATUS_TIME_LIMIT}
    status = status_map.get(int(res.status), STATUS_UNKNOWN)

    objective = None
    bound = None
    primal_viol = 0.0
    solcount = 0
    audit_gap = 0.0
    stationarity_abs = 0.0
    if status == STATUS_OPTIMAL:
        normalized_objective = float(res.fun) + objective_constant
        objective = normalized_objective
        normalized_bound, audit_gap, stationarity_abs = _audit_highs_dual(
            lp, res, obj, objective_constant, A_eq, b_eq, A_ub, b_ub)
        bound = normalized_bound
        if sense == "max":
            objective = -objective
            bound = None if bound is None else -bound
        if bound is None:
            status = STATUS_TIME_LIMIT
        # Feasibility of the returned primal on the ORIGINAL model.
        x = np.asarray(res.x, dtype=np.float64)
        Ax = A @ x
        viol = float(max(
            np.max(np.abs(np.clip(Ax, clb, cub) - Ax)) if Ax.size else 0.0,
            np.max(np.abs(np.clip(x, lp.variable_lower_bounds,
                                  lp.variable_upper_bounds) - x)) if x.size else 0.0,
        ))
        primal_viol = viol
        solcount = 1 if viol <= 1e-5 else 0
    elif status == STATUS_INFEASIBLE:
        bound = float("inf") if sense == "min" else -float("inf")

    return CertifiedSolve(
        status=status,
        objective=objective,
        bound=bound,
        bound_ok=bound is not None,
        audit_gap=audit_gap,
        stationarity_abs=stationarity_abs,
        primal_viol=primal_viol,
        solcount=solcount,
        time_sec=_time.time() - t0,
        num_iter=int(getattr(res, "nit", 0)),
        device="highs",
        x=(np.asarray(res.x, dtype=np.float64)
           if getattr(res, "x", None) is not None else None),
        termination=f"HIGHS_{int(res.status)}",
    )


# ---------------------------------------------------------------------------
# Phase 3: per-size backend recommendation (encoded from the benchmark
# findings; see docs/mlxpdlp_integration_plan.md section 15).
# ---------------------------------------------------------------------------

def recommend_solve_plan(num_variables: int, nnz: int,
                         need_certified: bool = True,
                         gurobi_available: bool = False) -> Dict:
    """Advisory backend plan for an LP of the given size.

    Based on the Phase 0/3 measurements on Apple Silicon (M3 Max):
      * tiny (<~1k vars): exact solvers win outright (gurobi restricted
        license fits; otherwise HiGHS).
      * small/medium (1k-30k vars): Metal FP32 with the opt-in 1e-5
        accuracy (host-FP64 correction) certifies well-conditioned LPs:
        medium in ~32s vs HiGHS ~36s and CPU@1e-6 139-547s; keep HiGHS
        as the fallback for ill-conditioned models.
      * large (30k+ vars): ladder metal@1e-5(+polish) -> cpu@1e-6 -
        HiGHS and gurobi(restricted) cannot solve these in budget.
      * need_certified=False: metal@1e-4 with the margin policy works
        from small up; the certified bound looseness is ~1e-3..6e-3
        relative.
    """
    if not need_certified:
        return {"device": "gpu" if num_variables > 1000 else
                ("gurobi" if gurobi_available else "highs"),
                "tol": 1e-4, "fallback": [],
                "note": "margin-policy fast path"}
    if num_variables <= 2000:
        return {"device": "gurobi" if gurobi_available else "highs",
                "tol": None, "fallback": [],
                "note": "exact solver wins at tiny/small sizes"}
    if num_variables <= 30000:
        return {"device": "gpu", "tol": 1e-5, "metal_polish": "auto",
                "fallback": ["highs"],
                "note": "metal@1e-5 + host-FP64 correction certifies "
                        "well-conditioned LPs (~32s at medium vs HiGHS "
                        "~36s); HiGHS fallback for ill-conditioned ones"}
    return {"device": "gpu", "tol": 1e-5, "metal_polish": "auto",
            "fallback": ["cpu"],
            "note": "metal@1e-5+polish -> cpu@1e-6 ladder "
                    "(HiGHS/gurobi-restricted cannot solve this size)"}


# ---------------------------------------------------------------------------
# Phase 4: variable-range prescaling (conditioning guard).
#
# Phase 3 showed PDLP stalls on badly scaled MODEL variables (the resnet
# unscaled-vs-scaled experiment: 2330s vs 2s) while objective scaling is
# harmless (Ruiz/Pock-Chambolle cover it). This preprocessor substitutes
# x_i = lb_i + r_i x'_i for variables with finite two-sided bounds,
# mapping every such variable into [0, 1]. It also shrinks the rigorous
# certificate correction (the scaled box has radius = n instead of the
# original range sum).
# ---------------------------------------------------------------------------

def prescale_lp_variables(lp: LPProblem, trigger_ratio: float = 1e6):
    """Return (scaled_lp, mapping) with finite two-sided variables mapped
    into [0,1]. Applied when the positive bound-range ratio exceeds
    `trigger_ratio` (auto conditioning guard).

    Transform: x_i = lb_i + r_i x'_i on the scaled indices (r_i = ub_i -
    lb_i > 0), other variables untouched. Variable ORDER is preserved
    (the constraint matrix is scaled column-wise by a diagonal matrix).

    mapping = (lb, d) with d_i = r_i on scaled indices, 1 elsewhere:
        x = lb + d * x'   (lb_i = 0 for unscaled indices).
    """
    lb = np.asarray(lp.variable_lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.variable_upper_bounds, dtype=np.float64)
    finite = np.isfinite(lb) & np.isfinite(ub)
    r = ub - lb
    pos = finite & (r > 0)
    if not pos.any():
        return None
    rmax = float(np.max(r[pos]))
    rmin = float(np.min(r[pos]))
    if rmax <= trigger_ratio * rmin:
        return None
    d = np.where(pos, r, 1.0)
    lb_shift = np.where(pos, lb, 0.0)

    A = lp.scipy_csr()
    A_scaled = A @ scipy_spdiag(d)          # columns scaled in place
    offset = np.asarray(A @ lb_shift).ravel()

    lp2 = lp.copy()
    lp2.row_ptr = np.asarray(A_scaled.indptr, dtype=np.int64)
    lp2.col_indices = np.asarray(A_scaled.indices, dtype=np.int64)
    lp2.values = np.asarray(A_scaled.data, dtype=np.float64)

    c2 = np.asarray(lp.objective, dtype=np.float64) * d
    lp2.objective = c2
    lp2.objective_constant = float(lp.objective_constant) + \
        float(np.dot(np.asarray(lp.objective, dtype=np.float64), lb_shift))

    lb2 = np.where(pos, 0.0, lb)
    ub2 = np.where(pos, 1.0, ub)
    lp2.variable_lower_bounds = lb2
    lp2.variable_upper_bounds = ub2

    clb = np.asarray(lp.constraint_lower_bounds, dtype=np.float64)
    cub = np.asarray(lp.constraint_upper_bounds, dtype=np.float64)
    lp2.constraint_lower_bounds = clb - offset
    lp2.constraint_upper_bounds = cub - offset
    return lp2, (lb_shift, d)


def map_scaled_result_to_original(lp, result2, mapping):
    """Map a SolveResult of the SCALED LP back to original coordinates.

    Returns a proxy object usable with audit_certificate(lp, proxy):
    primal_solution = lb + d * x', reduced_cost = z'/d (original var
    order, correct bound-dual signs), dual_solution unchanged, and the
    dual objective shifted by the objective offset c'lb.
    """
    lb_shift, d = mapping
    import types as _types
    x2 = np.asarray(result2.primal_solution, dtype=np.float64)
    z2 = np.asarray(result2.reduced_cost, dtype=np.float64)
    proxy = _types.SimpleNamespace(
        primal_solution=lb_shift + d * x2,
        dual_solution=np.asarray(result2.dual_solution, dtype=np.float64),
        reduced_cost=z2 / d,
        dual_objective_value=float(result2.dual_objective_value),
    )
    return proxy


def map_warm_start_to_scaled(warm_start, mapping):
    """Map an original-coordinate warm start into a prescaled LP.

    For ``x = lb + d*x'``, row duals are unchanged and stationarity gives
    ``z' = d*z`` for reduced costs.
    """
    lb_shift, d = mapping
    d = np.asarray(d, dtype=np.float64)
    scaled = {
        "primal": ((np.asarray(warm_start["primal"], dtype=np.float64) -
                    np.asarray(lb_shift, dtype=np.float64)) / d),
        "dual": np.asarray(warm_start["dual"], dtype=np.float64).copy(),
    }
    if warm_start.get("reduced_cost") is not None:
        scaled["reduced_cost"] = (
            d * np.asarray(warm_start["reduced_cost"], dtype=np.float64))
    return scaled


def scipy_spdiag(v):
    import scipy.sparse as sp
    return sp.diags(v, format="csr")
