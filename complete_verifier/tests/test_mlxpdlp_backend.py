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
"""Tests for the mlxPDLP backend bridge (Phase 0).

These tests are self-contained (no torch/gurobi needed) and skip when
mlxpdlp is not installed. Run e.g.:
    pytest complete_verifier/tests/test_mlxpdlp_backend.py -v
"""

import importlib.util
import os
import time
import types

import numpy as np
import pytest

_BACKEND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "lp_mip_solver", "mlxpdlp_backend.py")
_spec = importlib.util.spec_from_file_location("mlxpdlp_backend_test", _BACKEND_PATH)
_backend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend)
LPProblem = _backend.LPProblem
solve_mlxpdlp = _backend.solve_mlxpdlp
audit_certificate = _backend.audit_certificate
make_parameters = _backend.make_parameters

try:
    import mlxpdlp  # noqa: F401
    HAS_MLXPDLP = True
except ImportError:
    HAS_MLXPDLP = False

pytestmark = pytest.mark.skipif(not HAS_MLXPDLP, reason="mlxpdlp not installed")


def tiny_lp():
    """min -x0 - x1 s.t. x0 + x1 <= 1, x >= 0. Optimum: -1 at (1,0)/(0,1)."""
    return LPProblem(
        num_variables=2,
        num_constraints=1,
        row_ptr=np.array([0, 2], dtype=np.int64),
        col_indices=np.array([0, 1], dtype=np.int64),
        values=np.array([1.0, 1.0]),
        variable_lower_bounds=np.zeros(2),
        variable_upper_bounds=np.full(2, np.inf),
        constraint_lower_bounds=np.array([-np.inf]),
        constraint_upper_bounds=np.array([1.0]),
        objective=np.array([-1.0, -1.0]),
    )


def test_cpu_solve_optimal_and_certified():
    r = solve_mlxpdlp(tiny_lp(), device="cpu", tol=1e-7)
    assert r["status"] == "OPTIMAL"
    assert abs(r["objective"] - (-1.0)) < 1e-6
    assert r["cert_lb_ok"], f"certificate failed, audit_gap={r['audit_gap']}"
    assert r["cert_lb"] <= -1.0 + 1e-6, "dual certificate must be a valid lower bound"
    assert r["primal_viol"] < 1e-6


def test_gpu_solve_optimal_at_fp32_tolerance():
    if not mlxpdlp.has_gpu():
        pytest.skip("no Metal device")
    r = solve_mlxpdlp(tiny_lp(), device="gpu", tol=1e-4)
    assert r["status"] == "OPTIMAL"
    assert abs(r["objective"] - (-1.0)) < 1e-3
    # Dual certificate must never exceed the true optimum (sound direction):
    # for min, cert_lb <= opt. If FP32 audit fails, cert_lb is None (never
    # a wrong bound).
    if r["cert_lb_ok"]:
        assert r["cert_lb"] <= -1.0 + 1e-3


def test_audit_certificate_matches_analytic():
    lp = tiny_lp()
    r = solve_mlxpdlp(lp, device="cpu", tol=1e-8)
    audit = audit_certificate(lp, r["result"])
    # Stationarity on the original model: c - A'y - z -> 0.
    assert audit["audit_gap"] < 1e-6
    # Dual feasibility + dual objective => sound lower bound.
    assert audit["cert_lb"] <= -1.0 + 1e-6
    # Rigorous corrected bound must also be sound (x is unbounded above
    # here, so radius is infinite and safe_lb is None).
    assert audit["safe_lb"] is None


def test_safe_lb_sound_on_finite_box():
    # min -x0 - x1 s.t. x0 + x1 <= 1, 0 <= x <= 10. Optimum -1.
    lp = tiny_lp()
    lp.variable_upper_bounds = np.full(2, 10.0)
    r = solve_mlxpdlp(lp, device="cpu", tol=1e-4)
    audit = audit_certificate(lp, r["result"])
    assert audit["safe_lb"] is not None
    assert audit["safe_lb"] <= -1.0 + 1e-3, (
        f"safe_lb={audit['safe_lb']} exceeds the true optimum -1")
    # safe_lb may be looser than the raw dual objective, never tighter.
    assert audit["safe_lb"] <= float(r["result"].dual_objective_value) + 1e-12


def test_unbounded_stationarity_residual_never_certifies_raw_dual():
    """A tolerance-sized residual can make the raw dual objective unsafe
    when a variable is unbounded, so it must not be accepted by a gate."""
    # min x s.t. x >= -1e9, x free.  The true optimum is -1e9.
    lp = LPProblem(
        num_variables=1,
        num_constraints=1,
        row_ptr=np.array([0, 1], dtype=np.int64),
        col_indices=np.array([0], dtype=np.int64),
        values=np.array([1.0]),
        variable_lower_bounds=np.array([-np.inf]),
        variable_upper_bounds=np.array([np.inf]),
        constraint_lower_bounds=np.array([-1e9]),
        constraint_upper_bounds=np.array([np.inf]),
        objective=np.array([1.0]),
    )
    y = 1.0 - 5e-7
    fake = types.SimpleNamespace(
        primal_solution=np.array([-1e9]),
        dual_solution=np.array([y]),
        reduced_cost=np.array([0.0]),
        # This is 500 above the true optimum despite a 5e-7 residual.
        dual_objective_value=-1e9 * y,
    )
    audit = audit_certificate(lp, fake, cert_gate=1e-6)
    assert audit["audit_gap"] < 1e-6
    assert audit["safe_lb"] is None
    assert audit["cert_lb"] is None
    assert audit["cert_lb_ok"] is False


def test_audit_recomputes_dual_objective_instead_of_trusting_reported_value():
    lp = tiny_lp()
    lp.variable_upper_bounds = np.full(2, 10.0)
    fake = types.SimpleNamespace(
        primal_solution=np.array([1.0, 0.0]),
        dual_solution=np.array([-1.0]),
        reduced_cost=np.array([0.0, 0.0]),
        dual_objective_value=12345.0,
    )
    audit = audit_certificate(lp, fake)
    assert audit["dual_objective"] == pytest.approx(-1.0)
    assert audit["safe_lb"] == pytest.approx(-1.0)
    assert audit["safe_lb"] != fake.dual_objective_value


def test_warm_start_requires_presolve_off():
    lp = tiny_lp()
    warm = {"primal": np.zeros(2), "dual": np.zeros(1), "reduced_cost": np.zeros(2)}
    with pytest.raises(ValueError, match="presolve"):
        solve_mlxpdlp(lp, device="cpu", presolve=True, warm_start=warm)
    r = solve_mlxpdlp(lp, device="cpu", tol=1e-7, presolve=False, warm_start=warm)
    assert r["status"] == "OPTIMAL"
    assert abs(r["objective"] - (-1.0)) < 1e-6


def test_make_parameters_tolerance():
    p = make_parameters(tol=1e-6)
    assert p.termination_criteria.eps_optimal_relative == 1e-6
    assert p.termination_criteria.eps_feasible_relative == 1e-6


def test_objective_constant():
    lp = tiny_lp()
    lp.objective_constant = 0.5
    r = solve_mlxpdlp(lp, device="cpu", tol=1e-7)
    assert abs(r["objective"] - (-0.5)) < 1e-6


# ---------------------------------------------------------------------------
# gurobi_to_mlxpdlp with a mock gurobipy (no license needed)
# ---------------------------------------------------------------------------

class _MockGurobipy:
    """Minimal gurobipy stand-in implementing the API surface used by
    gurobi_to_mlxpdlp."""
    class _GRB:
        CONTINUOUS = "C"
        BINARY = "B"
        MINIMIZE = 1
        MAXIMIZE = -1
        EQUAL = "="
        LESS_EQUAL = "<"
        GREATER_EQUAL = ">"
        RANGE = "R"
        INFINITY = 1e30
    GRB = _GRB

    class Var:
        def __init__(self, name, lb, ub, obj, vtype="C"):
            self.VarName = name
            self.LB = lb
            self.UB = ub
            self.Obj = obj
            self.VType = vtype

    class Constr:
        def __init__(self, name, sense, rhs, sarhs=0.0):
            self.ConstrName = name
            self.Sense = sense
            self.RHS = rhs
            self.SARHSUp = sarhs

    class Model:
        def __init__(self, A, vars_, constrs, sense=1, objcon=0.0):
            self._A = A
            self._vars = vars_
            self._constrs = constrs
            self.ModelSense = sense
            self.ObjCon = objcon

        def getA(self):
            return self._A

        def getVars(self):
            return self._vars

        def getConstrs(self):
            return self._constrs


def test_gurobi_to_mlxpdlp_mock(monkeypatch):
    import scipy.sparse as sp
    grb = _MockGurobipy
    monkeypatch.setitem(__import__("sys").modules, "gurobipy", grb)

    # min  x0 - x1
    # s.t. x0 + x1 <= 5   (c0)
    #      x0 >= 1        (c1)
    #      x1 free-ish in [-10, 10]
    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]]))
    v0 = grb.Var("x0", 0.0, grb.GRB.INFINITY, 1.0)
    v1 = grb.Var("x1", -10.0, 10.0, -1.0)
    c0 = grb.Constr("c0", grb.GRB.LESS_EQUAL, 5.0)
    c1 = grb.Constr("c1", grb.GRB.GREATER_EQUAL, 1.0)
    model = grb.Model(A, [v0, v1], [c0, c1], sense=grb.GRB.MINIMIZE)

    lp = _backend.gurobi_to_mlxpdlp(model)
    assert lp.num_variables == 2 and lp.num_constraints == 2
    assert np.allclose(lp.objective, [1.0, -1.0])
    assert lp.variable_upper_bounds[0] == np.inf
    assert lp.variable_lower_bounds[1] == -10.0
    assert np.allclose(lp.constraint_upper_bounds, [5.0, np.inf])
    assert np.allclose(lp.constraint_lower_bounds, [-np.inf, 1.0])
    assert lp.variable_names == ["x0", "x1"]
    assert lp.constraint_names == ["c0", "c1"]


def test_gurobi_to_mlxpdlp_rejects_mip(monkeypatch):
    import scipy.sparse as sp
    grb = _MockGurobipy
    monkeypatch.setitem(__import__("sys").modules, "gurobipy", grb)
    A = sp.csr_matrix((0, 1))
    v0 = grb.Var("x0", 0.0, 1.0, 1.0, vtype=grb.GRB.BINARY)
    model = grb.Model(A, [v0], [], sense=grb.GRB.MINIMIZE)
    with pytest.raises(ValueError, match="non-continuous"):
        _backend.gurobi_to_mlxpdlp(model)


def test_gurobi_to_mlxpdlp_maximize_flips_objective(monkeypatch):
    import scipy.sparse as sp
    grb = _MockGurobipy
    monkeypatch.setitem(__import__("sys").modules, "gurobipy", grb)
    A = sp.csr_matrix((0, 1))
    v0 = grb.Var("x0", 0.0, 1.0, 2.0)
    model = grb.Model(A, [v0], [], sense=grb.GRB.MAXIMIZE, objcon=3.0)
    lp = _backend.gurobi_to_mlxpdlp(model)
    # maximize 2 x0 + 3  <=>  minimize -2 x0 - 3
    assert np.allclose(lp.objective, [-2.0])
    assert lp.objective_constant == -3.0


# ---------------------------------------------------------------------------
# Phase 1: certified wrapper tests
# ---------------------------------------------------------------------------

_STATUS = _backend.mlxpdlp_status_to_gurobi
_CertifiedSolve = _backend.CertifiedSolve


def test_status_mapping_by_name():
    assert _STATUS("OPTIMAL") == 2
    assert _STATUS("PRIMAL_INFEASIBLE") == 3
    assert _STATUS("DUAL_INFEASIBLE") == 5
    assert _STATUS("INFEASIBLE_OR_UNBOUNDED") == 4
    assert _STATUS("TIME_LIMIT") == 9
    assert _STATUS("ITERATION_LIMIT") == 9
    assert _STATUS("UNSPECIFIED") == -1
    assert _STATUS("FEAS_POLISH_SUCCESS") == 2


def test_status_mapping_by_enum():
    assert _STATUS(mlxpdlp.TerminationReason.OPTIMAL) == 2
    assert _STATUS(mlxpdlp.TerminationReason.PRIMAL_INFEASIBLE) == 3
    assert _STATUS(mlxpdlp.TerminationReason.TIME_LIMIT) == 9


def test_solve_certified_min_sound():
    cs = _backend.solve_certified(tiny_lp(), device="cpu", tol=1e-7)
    assert cs.status == 2, cs.termination
    # Certified lower bound never exceeds the true optimum -1.
    assert cs.bound <= -1.0 + 1e-6
    assert cs.objective == pytest.approx(-1.0, abs=1e-6)
    assert cs.objective >= cs.bound  # primal incumbent vs dual lower bound


def test_solve_certified_max_sound():
    # maximize x0 + x1 s.t. x0 + x1 <= 1, 0 <= x <= 10  -> optimum +1
    lp = tiny_lp()
    lp.variable_upper_bounds = np.full(2, 10.0)
    lp.objective = np.array([1.0, 1.0])
    cs = _backend.solve_certified(lp, device="cpu", tol=1e-7, sense="max")
    assert cs.status == 2, cs.termination
    # Certified upper bound never underestimates the true optimum +1.
    assert cs.bound >= 1.0 - 1e-6
    assert abs(cs.objective - 1.0) < 1e-6


def _fake_certified(status=9, bound=None, bound_ok=False, objective=None,
                    solcount=0, device="fake"):
    return _CertifiedSolve(
        status=status, objective=objective, bound=bound, bound_ok=bound_ok,
        audit_gap=1e-4, stationarity_abs=1e-4, primal_viol=0.0,
        time_sec=0.0, num_iter=0, device=device, termination="FAKE",
        solcount=solcount)


def test_margin_fast_path_accepts_sound_corrected_bound(monkeypatch):
    calls = []

    def fast(*args, **kwargs):
        return _fake_certified(status=9, bound=0.5, bound_ok=True,
                               device="gpu")

    def fallback(lp):
        calls.append("fallback")
        return _fake_certified(status=2, bound=1.0, bound_ok=True)

    monkeypatch.setattr(_backend, "_solve_one", fast)
    cs = _backend.solve_certified(
        tiny_lp(), margin=0.1, threshold=0.0, fallback=[fallback])
    assert calls == []
    assert cs.device == "gpu" and cs.bound == 0.5 and cs.bound_ok


def test_unknown_termination_is_not_promoted_to_optimal(monkeypatch):
    result = types.SimpleNamespace(
        termination_reason="UNSPECIFIED",
        termination_reason_name="UNSPECIFIED",
        primal_solution=np.array([0.0, 0.0]),
    )
    monkeypatch.setattr(
        _backend, "solve_mlxpdlp",
        lambda *a, **k: {
            "result": result, "objective": 0.0, "time_sec": 0.0,
            "num_iter": 0,
        })
    monkeypatch.setattr(
        _backend, "audit_certificate",
        lambda *a, **k: {
            "cert_lb_ok": True, "cert_lb": -1.0, "safe_lb": None,
            "audit_gap": 0.0, "stationarity_abs": 0.0,
            "primal_viol": 1.0,
        })
    cs = _backend._solve_one(tiny_lp(), "cpu", 1e-6, 1.0, "min")
    assert cs.status == _backend.STATUS_UNKNOWN
    assert cs.bound_ok and cs.bound == -1.0
    assert cs.solcount == 0


def test_ladder_uses_one_total_time_budget(monkeypatch):
    seen = []

    def first(*args, **kwargs):
        time.sleep(0.02)
        return _fake_certified()

    def highs(lp, time_limit, sense):
        seen.append(time_limit)
        return _fake_certified(status=2, bound=0.0, bound_ok=True,
                               device="highs")

    monkeypatch.setattr(_backend, "_solve_one", first)
    monkeypatch.setattr(_backend, "solve_highs_lp", highs)
    _backend.solve_certified(tiny_lp(), time_limit=0.1, fallback=["highs"])
    assert len(seen) == 1
    assert 0.0 < seen[0] < 0.095


def test_expired_total_budget_does_not_start_solver(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("solver started after its deadline")

    monkeypatch.setattr(_backend, "_solve_one", should_not_run)
    cs = _backend.solve_certified(tiny_lp(), time_limit=0.0, fallback=[])
    assert cs.status == _backend.STATUS_TIME_LIMIT
    assert cs.bound is None and cs.solcount == 0


def test_margin_ladder_escalates_and_uses_fallback():
    calls = []

    def fake_fallback(lp):
        calls.append("fake")
        return _CertifiedSolve(status=2, objective=100.0, bound=100.0,
                               bound_ok=True, audit_gap=0.0,
                               stationarity_abs=0.0, primal_viol=0.0,
                               time_sec=0.0, num_iter=0, device="fake",
                               termination="OPTIMAL")

    cs = _backend.solve_certified(
        tiny_lp(), device="cpu", tol=1e-7, sense="min",
        margin=50.0, threshold=0.0, fallback=[fake_fallback])
    assert calls == ["fake"], "ladder must escalate when the bound is not decisive"
    assert cs.bound == 100.0 and cs.device == "fake"


def test_margin_ladder_skips_fallback_when_decisive():
    calls = []

    def fake_fallback(lp):
        calls.append("fake")
        return _CertifiedSolve(status=2, objective=0.0, bound=0.0,
                               bound_ok=True, audit_gap=0.0,
                               stationarity_abs=0.0, primal_viol=0.0,
                               time_sec=0.0, num_iter=0, device="fake",
                               termination="OPTIMAL")

    # bound -1 >= threshold -2 + margin 0.5 -> decisive, no escalation.
    cs = _backend.solve_certified(
        tiny_lp(), device="cpu", tol=1e-7, sense="min",
        margin=0.5, threshold=-2.0, fallback=[fake_fallback])
    assert calls == []
    assert cs.status == 2 and cs.bound <= -1.0 + 1e-6


def test_ladder_survives_missing_gurobi():
    # gurobipy is not importable in this env: the ladder must swallow the
    # failure and return the first-stage result.
    import builtins
    real_import = builtins.__import__

    def no_gurobi(name, *a, **k):
        if name == "gurobipy":
            raise ImportError("no gurobipy")
        return real_import(name, *a, **k)

    try:
        builtins.__import__ = no_gurobi
        cs = _backend.solve_certified(tiny_lp(), device="cpu", tol=1e-7,
                                      margin=50.0, threshold=0.0,
                                      fallback="gurobi")
    finally:
        builtins.__import__ = real_import
    assert cs.status == 2
    assert cs.bound <= -1.0 + 1e-6  # first-stage CPU result, still certified


def test_decide_safe_logic():
    def cs(status, bound=None, objective=None, viol=0.0):
        return _CertifiedSolve(status=status, objective=objective, bound=bound,
                               bound_ok=status == 2, audit_gap=0.0,
                               stationarity_abs=0.0, primal_viol=viol,
                               time_sec=0.0, num_iter=0, device="t",
                               termination="t")

    # min sense, threshold 0
    assert _backend.decide_safe(cs(3), 0.0) == "safe"                    # infeasible
    assert _backend.decide_safe(cs(2, bound=0.5), 0.0, margin=0.1) == "safe"
    assert _backend.decide_safe(cs(2, bound=0.05), 0.0, margin=0.1) == "unknown"
    assert _backend.decide_safe(cs(9, bound=0.5), 0.0, margin=0.1) == "safe"  # TIME_LIMIT but certified
    assert _backend.decide_safe(cs(2, bound=None, objective=-1.0, viol=0.0),
                                0.0, margin=0.1) == "unsafe"             # feasible witness
    assert _backend.decide_safe(cs(2, bound=None, objective=-1.0, viol=1e-3),
                                0.0, margin=0.1) == "unknown"            # infeasible witness
    assert _backend.decide_safe(cs(9, bound=None, objective=None), 0.0) == "unknown"
    # max sense, threshold 0
    assert _backend.decide_safe(cs(2, bound=-0.5), 0.0, margin=0.1, sense="max") == "safe"
    assert _backend.decide_safe(cs(2, bound=None, objective=0.5, viol=0.0),
                                0.0, margin=0.1, sense="max") == "unsafe"


def test_counterexample_feasible():
    r = solve_mlxpdlp(tiny_lp(), device="cpu", tol=1e-8)
    assert _backend.counterexample_feasible(tiny_lp(), r["result"]) is True

    # Artificially violated primal solution: x0 = 5 (violates x0 + x1 <= 1).
    fake = types.SimpleNamespace(
        primal_solution=np.array([5.0, 0.0]),
        dual_solution=np.array([0.0]),
        reduced_cost=np.array([0.0, 0.0]),
        dual_objective_value=-10.0,
    )
    assert _backend.counterexample_feasible(tiny_lp(), fake) is False


def _load_probe():
    import importlib.util
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "lp_mip_solver", "probe_mlxpdlp.py")
    spec = importlib.util.spec_from_file_location("probe_mlxpdlp_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_never_unsound_random_networks():
    """Property test: on random small ReLU-MLP relaxations, the certified
    bound never exceeds the exact HiGHS optimum (min sense), on both CPU
    and Metal."""
    scipy = pytest.importorskip("scipy")
    probe = _load_probe()
    for seed in range(20):
        lp, info = probe.build_relu_mlp_lp((8, 16, 16, 4), seed=seed)
        h = probe.solve_highs(lp)
        if not h["success"]:
            continue
        opt = h["objective"]
        cs = _backend.solve_certified(lp, device="cpu", tol=1e-5)
        if cs.bound is None:
            continue
        assert cs.bound <= opt + 1e-6, (
            f"seed {seed}: certified CPU bound {cs.bound} exceeds optimum {opt}")
    if mlxpdlp.has_gpu():
        for seed in range(8):
            lp, info = probe.build_relu_mlp_lp((8, 16, 16, 4), seed=seed)
            h = probe.solve_highs(lp)
            if not h["success"]:
                continue
            opt = h["objective"]
            cs = _backend.solve_certified(lp, device="gpu", tol=1e-4)
            if cs.bound is None:
                continue
            assert cs.bound <= opt + 1e-4, (
                f"seed {seed}: certified Metal bound {cs.bound} exceeds optimum {opt}")


# ---------------------------------------------------------------------------
# HiGHS fallback / reference
# ---------------------------------------------------------------------------

def test_solve_highs_lp_optimal():
    cs = _backend.solve_highs_lp(tiny_lp())
    assert cs.status == 2, cs.termination
    assert abs(cs.objective - (-1.0)) < 1e-6
    assert cs.bound <= -1.0 + 1e-6
    assert cs.device == "highs" and cs.solcount == 1


def test_solve_highs_lp_preserves_objective_constants_and_sense():
    lp = LPProblem(
        num_variables=1, num_constraints=0,
        row_ptr=np.array([0], dtype=np.int64),
        col_indices=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float64),
        variable_lower_bounds=np.array([0.0]),
        variable_upper_bounds=np.array([1.0]),
        constraint_lower_bounds=np.array([], dtype=np.float64),
        constraint_upper_bounds=np.array([], dtype=np.float64),
        objective=np.array([2.0]), objective_constant=5.0,
    )
    mn = _backend.solve_highs_lp(lp, sense="min")
    mx = _backend.solve_highs_lp(lp, sense="max")
    assert mn.objective == pytest.approx(5.0)
    assert mn.bound <= 5.0
    assert mx.objective == pytest.approx(7.0)
    assert mx.bound >= 7.0


def test_highs_bound_comes_from_audited_dual_not_primal(monkeypatch):
    scipy = pytest.importorskip("scipy")
    lp = LPProblem(
        num_variables=1, num_constraints=0,
        row_ptr=np.array([0], dtype=np.int64),
        col_indices=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float64),
        variable_lower_bounds=np.array([0.0]),
        variable_upper_bounds=np.array([1.0]),
        constraint_lower_bounds=np.array([], dtype=np.float64),
        constraint_upper_bounds=np.array([], dtype=np.float64),
        objective=np.array([1.0]),
    )
    empty = types.SimpleNamespace(marginals=np.array([], dtype=np.float64))
    fake = types.SimpleNamespace(
        status=0, fun=0.5, x=np.array([0.5]), nit=1,
        eqlin=empty, ineqlin=empty,
        lower=types.SimpleNamespace(marginals=np.array([1.0])),
        upper=types.SimpleNamespace(marginals=np.array([0.0])),
    )
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **k: fake)
    cs = _backend.solve_highs_lp(lp)
    assert cs.objective == pytest.approx(0.5)
    assert cs.bound == pytest.approx(0.0)
    assert cs.bound != cs.objective


def test_solve_highs_lp_infeasible():
    lp = tiny_lp()
    lp.constraint_upper_bounds = np.array([-1.0])  # x0 + x1 <= -1, x >= 0
    cs = _backend.solve_highs_lp(lp)
    assert cs.status == 3


def test_ladder_highs_fallback():
    cs = _backend.solve_certified(tiny_lp(), device="cpu", tol=1e-7,
                                  margin=50.0, threshold=0.0,
                                  fallback="highs")
    assert cs.device == "highs"
    assert cs.status == 2
    assert cs.bound <= -1.0 + 1e-6


def test_ladder_gurobi_fallback_restricted_license():
    """gurobipy 13+ ships a restricted non-production license that can
    solve small models; the ladder must use it when available."""
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        pytest.skip("gurobipy not installed")
    try:
        cs = _backend.solve_certified(tiny_lp(), device="cpu", tol=1e-7,
                                      margin=50.0, threshold=0.0,
                                      fallback="gurobi")
    except Exception:
        pytest.skip("gurobi restricted license unavailable in this env")
    assert cs.device == "gurobi"
    assert cs.status == 2
    assert cs.bound <= -1.0 + 1e-6


def test_solve_gurobi_lp_preserves_constants_and_uses_dual_bound():
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        pytest.skip("gurobipy not installed")
    lp = LPProblem(
        num_variables=1, num_constraints=0,
        row_ptr=np.array([0], dtype=np.int64),
        col_indices=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float64),
        variable_lower_bounds=np.array([0.0]),
        variable_upper_bounds=np.array([1.0]),
        constraint_lower_bounds=np.array([], dtype=np.float64),
        constraint_upper_bounds=np.array([], dtype=np.float64),
        objective=np.array([2.0]), objective_constant=5.0,
    )
    try:
        mn = _backend.solve_gurobi_lp(lp, sense="min")
        mx = _backend.solve_gurobi_lp(lp, sense="max")
    except Exception:
        pytest.skip("gurobi restricted license unavailable in this env")
    assert mn.objective == pytest.approx(5.0)
    assert mn.bound <= mn.objective
    assert mx.objective == pytest.approx(7.0)
    assert mx.bound >= mx.objective


# ---------------------------------------------------------------------------
# Phase 3: per-size recommendation policy
# ---------------------------------------------------------------------------

def test_recommend_solve_plan():
    p = _backend.recommend_solve_plan(100, 2000, need_certified=True,
                                      gurobi_available=False)
    assert p["device"] == "highs"
    p = _backend.recommend_solve_plan(1000, 100_000, need_certified=True,
                                      gurobi_available=True)
    assert p["device"] == "gurobi"
    p = _backend.recommend_solve_plan(10_000, 2_000_000, need_certified=True)
    assert p["device"] == "gpu" and p["fallback"] == ["highs"]
    p = _backend.recommend_solve_plan(50_000, 5_000_000, need_certified=True)
    assert p["device"] == "gpu" and p["fallback"] == ["cpu"]
    p = _backend.recommend_solve_plan(50_000, 5_000_000, need_certified=False)
    assert p["device"] == "gpu"


# ---------------------------------------------------------------------------
# Phase 4: prescaling, tuning knobs, Gurobi regression
# ---------------------------------------------------------------------------

def test_prescale_trigger_and_soundness():
    # Mixed-range LP: x0 in [0, 10], x1 in [0, 1e8]; min x0 s.t. x0 + 1e-8 x1 <= 1.
    lp = _backend.LPProblem(
        num_variables=2, num_constraints=1,
        row_ptr=np.array([0, 2], dtype=np.int64),
        col_indices=np.array([0, 1], dtype=np.int64),
        values=np.array([1.0, 1e-8]),
        variable_lower_bounds=np.zeros(2),
        variable_upper_bounds=np.array([10.0, 1e8]),
        constraint_lower_bounds=np.array([-np.inf]),
        constraint_upper_bounds=np.array([1.0]),
        objective=np.array([1.0, 0.0]),
    )
    prescaled = _backend.prescale_lp_variables(lp)
    assert prescaled is not None  # range ratio 1e7 > 1e6
    lp2, mapping = prescaled
    # transformed problem must have the same optimum (0.0)
    h2 = _backend.solve_highs_lp(lp2)
    assert h2.status == 2 and abs(h2.objective) < 1e-6
    # end-to-end: solve with prescale and audit on the ORIGINAL model
    out = _backend.solve_mlxpdlp(lp, device="cpu", tol=1e-5, prescale="auto")
    assert out["status"] == "OPTIMAL"
    assert abs(out["objective"]) < 1e-4
    # returned primal must be in ORIGINAL coordinates
    x = np.asarray(out["result"].primal_solution)
    assert x[0] < 1e-3 and x[1] < 1e5
    # certified bound sound: <= HiGHS optimum of the original LP
    h = _backend.solve_highs_lp(lp)
    assert out["cert_lb"] is not None
    assert out["cert_lb"] <= h.objective + 1e-6


def test_prescale_no_trigger_for_balanced_ranges():
    lp = tiny_lp()
    lp.variable_upper_bounds = np.full(2, 10.0)
    assert _backend.prescale_lp_variables(lp) is None


def test_prescale_soundness_random_mixed_scale():
    rng = np.random.default_rng(123)
    for seed in range(6):
        n = 6
        ranges = 10.0 ** rng.uniform(-3, 3, n)      # ranges 1e-3 .. 1e3
        lb = rng.uniform(-2, 2, n)
        ub = lb + ranges
        # random constraint rows
        m = 3
        A = rng.uniform(-1, 1, (m, n)) * 10.0 ** rng.uniform(-1, 1, (m, n))
        lp = _backend.LPProblem(
            num_variables=n, num_constraints=m,
            row_ptr=None, col_indices=None, values=None,
            objective=rng.uniform(-1, 1, n),
            variable_lower_bounds=lb, variable_upper_bounds=ub,
        )
        csr = lp.scipy_csr() if False else None
        import scipy.sparse as sp
        lp = _backend.LPProblem(
            num_variables=n, num_constraints=m,
            row_ptr=np.asarray(sp.csr_matrix(A).indptr, dtype=np.int64),
            col_indices=np.asarray(sp.csr_matrix(A).indices, dtype=np.int64),
            values=np.asarray(sp.csr_matrix(A).data, dtype=np.float64),
            objective=rng.uniform(-1, 1, n),
            variable_lower_bounds=lb, variable_upper_bounds=ub,
            constraint_lower_bounds=np.full(m, -np.inf),
            constraint_upper_bounds=rng.uniform(0, 2, m),
        )
        h = _backend.solve_highs_lp(lp)
        if h.status != 2:
            continue
        out = _backend.solve_mlxpdlp(lp, device="cpu", tol=1e-5,
                                     prescale="auto")
        if out["cert_lb"] is not None:
            assert out["cert_lb"] <= h.objective + 1e-4, (
                f"seed {seed}: cert {out['cert_lb']} > opt {h.objective}")


def test_make_parameters_tuning_knobs():
    p = _backend.make_parameters(tol=1e-5, ruiz_iterations=0, restart_policy=1)
    assert p.l_inf_ruiz_iterations == 0
    assert p.restart_policy == 1


def test_settings_knob_defaults_and_config():
    import sys as _sys
    fake_args = _sys.modules.get("arguments")
    s = _backend.get_lp_backend_settings()
    # defaults (no verifier config): ruiz=0 per Phase 3 finding
    assert s["ruiz_iterations"] == 0 and s["restart_policy"] == 0


def _load_probe_p4():
    import importlib.util
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "lp_mip_solver", "probe_mlxpdlp.py")
    spec = importlib.util.spec_from_file_location("probe_mlxpdlp_p4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gurobi_regression_tiny_net():
    """mlxPDLP certified bounds vs the Gurobi LP optimum (restricted
    license fits the tiny net) - the Phase 4 regression test."""
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        pytest.skip("gurobipy not installed")
    probe = _load_probe_p4()
    lp, info = probe.build_relu_mlp_lp(probe.SIZES["tiny"], seed=0)
    for oi in (0, 2):
        lp2 = probe.objective_on_variable(lp, info["y_out_range"][0] + oi)
        try:
            g = _backend.solve_gurobi_lp(lp2, time_limit=60)
        except Exception:
            pytest.skip("gurobi restricted license unavailable")
        assert g.status == 2
        cs = _backend.solve_certified(lp2, device="cpu", tol=1e-6,
                                      fallback=[])
        assert cs.status == 2
        # certified bound never exceeds the exact Gurobi optimum, and is
        # within 1e-3 relative of it
        assert cs.bound <= g.objective + 1e-6
        assert abs(cs.bound - g.objective) <= 1e-3 * max(1.0, abs(g.objective))


# ---------------------------------------------------------------------------
# Opt-in 1e-5 Metal accuracy (mlxPDLP >= 2026-08-23)
# ---------------------------------------------------------------------------

def test_metal_polish_enabled_logic():
    assert _backend.metal_polish_enabled(
        {"device": "gpu", "tol": 1e-4, "metal_polish": "on"}) is True
    assert _backend.metal_polish_enabled(
        {"device": "gpu", "tol": 1e-5, "metal_polish": "off"}) is False
    # auto: polish only on Metal at tolerance <= 1e-5
    assert _backend.metal_polish_enabled(
        {"device": "gpu", "tol": 1e-5, "metal_polish": "auto"}) is True
    assert _backend.metal_polish_enabled(
        {"device": "gpu", "tol": 1e-4, "metal_polish": "auto"}) is False
    assert _backend.metal_polish_enabled(
        {"device": "cpu", "tol": 1e-5, "metal_polish": "auto"}) is False


def test_settings_polish_parse():
    import sys as _sys
    fake_args = _sys.modules.get("arguments")
    s = _backend.get_lp_backend_settings()
    assert s["metal_polish"] in ("off", "on", "auto")


def test_metal_1e5_accuracy_regression():
    """The new mlxPDLP build reaches an audited 1e-5 on well-conditioned
    LPs: Metal@1e-5 + host polish must certify the tiny LP to 1e-5."""
    if not mlxpdlp.has_gpu():
        pytest.skip("no Metal device")
    lp = tiny_lp()
    cs = _backend.solve_certified(lp, device="gpu", tol=1e-5,
                                  host_polish=True, fallback=[])
    assert cs.status == 2, cs.termination
    assert cs.bound <= -1.0 + 1e-5, "certified bound must be sound"
    assert cs.bound >= -1.0 - 1e-4, "bound should reach ~1e-5 accuracy"


def test_cert_gate_tolerance_aware():
    # The tolerance gate is a convergence diagnostic, never permission to
    # accept an uncorrected residual as a proof.
    lp = tiny_lp()
    r = solve_mlxpdlp(lp, device="cpu", tol=1e-8)
    audit_strict = _backend.audit_certificate(lp, r["result"], cert_gate=1e-6)
    assert audit_strict["cert_lb_ok"] is True  # tight solve passes strict gate
    # simulate a 3e-6-gap certificate by adding stationarity noise
    proxy = types.SimpleNamespace(
        primal_solution=r["result"].primal_solution,
        dual_solution=r["result"].dual_solution,
        reduced_cost=np.asarray(r["result"].reduced_cost) + 3e-6,
        dual_objective_value=r["result"].dual_objective_value,
    )
    a1 = _backend.audit_certificate(lp, proxy, cert_gate=1e-6)
    assert a1["audit_tight"] is False
    assert a1["cert_lb_ok"] is False
    a2 = _backend.audit_certificate(lp, proxy, cert_gate=1e-5)
    assert a2["audit_tight"] is True
    assert a2["cert_lb_ok"] is False
