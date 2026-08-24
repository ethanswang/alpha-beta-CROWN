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
"""Phase 2 tests: mlxPDLP backend wired into the LP call sites.

Runs the REAL wired functions (lp_solver, all_node_split_LP,
mip_solver_lb_ub, mip_solver_lb_ub_and, solve_gurobi_model_with_mlxpdlp)
against a FakeGurobiModel that implements the gurobipy surface used by
the backend conversion. No Gurobi license needed; gurobipy is stubbed.

Requires torch (repo import chain) and mlxpdlp; skips otherwise.
"""

import importlib.util
import os
import sys
import time
import types

import numpy as np
import pytest

# Repo layout: the tests normally live under <repo>/complete_verifier/tests.
# ABCROWN_COMPLETE_VERIFIER_DIR overrides the complete_verifier directory,
# which lets the suite run from a copy outside the repo tree (the repo
# complete_verifier/__init__ pulls in onnx2pytorch and friends).
_REPO = os.environ.get(
    "ABCROWN_COMPLETE_VERIFIER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO)
_REPO_ROOT = os.path.dirname(_REPO)
sys.path.insert(0, os.path.join(_REPO_ROOT, "auto_LiRPA"))

# ---------------------------------------------------------------------------
# Stub gurobipy BEFORE the lp_mip_solver package is imported.
# ---------------------------------------------------------------------------

class _FakeEnv:
    def __init__(self, empty=False):
        pass
    def start(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def setParam(self, *a, **k):
        pass


class _GRBNamespace:
    """Permissive constants namespace: unknown attrs return a sentinel int."""
    CONTINUOUS = "C"
    BINARY = "B"
    INTEGER = "I"
    MINIMIZE = 1
    MAXIMIZE = -1
    EQUAL = "="
    LESS_EQUAL = "<"
    GREATER_EQUAL = ">"
    RANGE = "R"
    INFINITY = 1e30
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    UNBOUNDED = 5
    TIME_LIMIT = 9
    INTERRUPTED = 11
    def __getattr__(self, name):
        return 2


class _FakeGurobiError(Exception):
    pass


class _FakeLinExpr:
    """Row expression produced by FakeVar comparisons (v <= rhs)."""
    def __init__(self, terms, sense, rhs):
        self.terms = dict(terms)
        self.sense = sense
        self.rhs = float(rhs)
    def __ge__(self, other):
        return _FakeLinExpr(self.terms, ">", float(other))
    def __le__(self, other):
        return _FakeLinExpr(self.terms, "<", float(other))


class _FakeGurobipy(types.ModuleType):
    def __init__(self):
        super().__init__("gurobipy")
        self.GRB = _GRBNamespace()
        self.GurobiError = _FakeGurobiError
        self.Env = _FakeEnv
        self.LinExpr = _FakeLinExpr
        self.QuadExpr = _FakeLinExpr
        self.Model = None  # not used by the tested paths


# Install the fake gurobipy ONLY when the real one is unavailable.
# pytest imports all test modules at collection time, so a unconditional
# stub here would poison every other test file in the same process.
try:
    import gurobipy  # noqa: F401
except ImportError:
    sys.modules["gurobipy"] = _FakeGurobipy()


# ---------------------------------------------------------------------------
# Fake gurobi model wrapping an LPProblem (enough API for the backend
# conversion + the call-site operations the backend branch performs).
# ---------------------------------------------------------------------------

class FakeVar:
    VType = "C"
    def __init__(self, name, lb, ub, obj=0.0):
        self._name = name
        self._lb = float(lb)
        self._ub = float(ub)
        self._obj = float(obj)
    @property
    def VarName(self):
        return self._name
    @property
    def LB(self):
        return self._lb
    @property
    def UB(self):
        return self._ub
    @property
    def Obj(self):
        return self._obj
    @LB.setter
    def LB(self, v):
        self._lb = float(v)
    @UB.setter
    def UB(self, v):
        self._ub = float(v)
    @property
    def lb(self):
        return self._lb
    @lb.setter
    def lb(self, v):
        self._lb = float(v)
    @property
    def ub(self):
        return self._ub
    @ub.setter
    def ub(self, v):
        self._ub = float(v)
    def __le__(self, other):
        return _FakeLinExpr({self: 1.0}, "<", other)
    def __ge__(self, other):
        return _FakeLinExpr({self: 1.0}, ">", other)
    def __repr__(self):
        return f"FakeVar({self._name})"


class FakeConstr:
    def __init__(self, name, sense, rhs, sarhs=0.0):
        self.ConstrName = name
        self.Sense = sense
        self.RHS = float(rhs)
        self.SARHSUp = float(sarhs)


class FakeGurobiModel:
    """Mutable gurobi-like model: getA/getVars/getConstrs + addConstr +
    setObjective + getVarByName + copy."""

    def __init__(self, lp):
        self.num_variables = lp.num_variables
        self._vars = []
        for i in range(lp.num_variables):
            lb = lp.variable_lower_bounds[i] if lp.variable_lower_bounds is not None else -1e30
            ub = lp.variable_upper_bounds[i] if lp.variable_upper_bounds is not None else 1e30
            obj = lp.objective[i]
            name = (lp.variable_names[i] if lp.variable_names else f"var_{i}")
            self._vars.append(FakeVar(name, lb, ub, obj))
        A = lp.scipy_csr()
        self._rows = []  # (indices, coeffs, lb, ub)
        self._constrs = []
        for i in range(lp.num_constraints):
            row = A.getrow(i)
            inds = row.indices.tolist()
            coeffs = row.data.tolist()
            clb = lp.constraint_lower_bounds[i]
            cub = lp.constraint_upper_bounds[i]
            sense = "=" if clb == cub else ("<" if np.isfinite(cub) else ">")
            self._rows.append((inds, coeffs, clb, cub))
            name = lp.constraint_names[i] if lp.constraint_names else f"c{i}"
            self._constrs.append(FakeConstr(name, sense,
                                            cub if np.isfinite(cub) else clb))
        self.ModelSense = 1
        self.ObjCon = 0.0
        self.status = 0
        self._obj_var = None

    def getA(self):
        import scipy.sparse as sp
        if not self._rows:
            return sp.csr_matrix((0, self.num_variables))
        indptr = [0]
        cols, vals = [], []
        for r_indices, r_coeffs, _lb, _ub in self._rows:
            for j, c in zip(r_indices, r_coeffs):
                cols.append(j)
                vals.append(c)
            indptr.append(len(cols))
        return sp.csr_matrix((vals, cols, indptr),
                             shape=(len(self._rows), self.num_variables))

    def getVars(self):
        return list(self._vars)

    def getConstrs(self):
        return list(self._constrs)

    def getVarByName(self, name):
        for v in self._vars:
            if v.VarName == name:
                return v
        return None

    def addConstr(self, expr, name=None):
        assert isinstance(expr, _FakeLinExpr), type(expr)
        lb, ub = -np.inf, np.inf
        if expr.sense == "<":
            ub = expr.rhs
        elif expr.sense == ">":
            lb = expr.rhs
        else:
            lb = ub = expr.rhs
        idx_map = {v: i for i, v in enumerate(self._vars)}
        terms = {idx_map[v]: c for v, c in expr.terms.items()
                 if c != 0 and v in idx_map}
        inds = sorted(terms)
        coeffs = [terms[i] for i in inds]
        self._rows.append((inds, coeffs, lb, ub))
        sense = "<" if np.isfinite(ub) else ">"
        c = FakeConstr(name or f"c{len(self._constrs)}", sense,
                       ub if np.isfinite(ub) else lb)
        self._constrs.append(c)
        return c

    def setObjective(self, expr, sense=1):
        for v in self._vars:
            v._obj = 0.0
        if isinstance(expr, FakeVar):
            expr._obj = 1.0
            self._obj_var = expr
        self.ModelSense = 1
        self.ObjCon = 0.0

    def update(self):
        pass

    def copy(self):
        import copy as _copy
        return _copy.deepcopy(self)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BACKEND = _load(os.path.join(_REPO, "lp_mip_solver",
                              "mlxpdlp_backend.py"), "mlxpdlp_backend_p2")
_PROBE = _load(os.path.join(_REPO, "lp_mip_solver",
                            "probe_mlxpdlp.py"), "probe_mlxpdlp_p2")

pytest.importorskip("mlxpdlp")
torch = pytest.importorskip("torch")

# Import the REAL wired modules BY FILE PATH under their package names.
# This bypasses lp_mip_solver/__init__.py, whose import chain pulls in
# attack/load_model/onnx2pytorch etc. Relative imports inside the loaded
# modules resolve through the registered sys.modules entries.
_pkg = types.ModuleType("lp_mip_solver")
_pkg.__path__ = [os.path.join(_REPO, "lp_mip_solver")]
sys.modules["lp_mip_solver"] = _pkg

sys.modules["lp_mip_solver.mlxpdlp_backend"] = _BACKEND

solver_utils = _load(os.path.join(_REPO, "lp_mip_solver", "utils.py"),
                     "lp_mip_solver.utils")
sys.modules["lp_mip_solver.utils"] = solver_utils

bounds_core = _load(os.path.join(_REPO, "lp_mip_solver", "bounds_core.py"),
                    "lp_mip_solver.bounds_core")
sys.modules["lp_mip_solver.bounds_core"] = bounds_core

GurobiLikeResult = _BACKEND.GurobiLikeResult

SETTINGS_CPU = {
    "backend": "mlxpdlp", "device": "cpu", "tol": 1e-5,
    "margin": 1e-3, "fallback": None, "time_limit": 120.0,
    "ruiz_iterations": 0, "restart_policy": 0,
}


def make_net_lp(sizes=(8, 16, 16, 4), seed=0, objective_index=0):
    """Build a tiny network LP and its fake gurobi model."""
    lp, info = _PROBE.build_relu_mlp_lp(sizes, seed=seed)
    fake = FakeGurobiModel(lp)
    out_var = fake.getVarByName(
        f"var_{info['y_out_range'][0] + objective_index}")
    return lp, info, fake, out_var


class TestGurobiModelBackend:
    def test_min_max_certified_on_fake_model(self):
        lp, info, fake, out_var = make_net_lp()
        hmin = _PROBE.solve_highs(lp)
        assert hmin["success"]
        r = _BACKEND.solve_gurobi_model_with_mlxpdlp(
            fake, objective_var=out_var, sense="min",
            device="cpu", tol=1e-5, time_limit=60)
        assert r.status == 2, r.termination
        assert r.objbound <= hmin["objective"] + 1e-6  # certified: sound
        assert r.x[out_var.VarName] == pytest.approx(r.objval)
        assert r.objbound <= r.objval
        # max direction on the same variable
        lp_max, info2, fake2, out_var2 = make_net_lp(seed=0)
        hmax = _PROBE.solve_highs(_PROBE.objective_on_variable(lp_max, 0, maximize=True))
        r2 = _BACKEND.solve_gurobi_model_with_mlxpdlp(
            fake2, objective_var=out_var2, sense="max",
            device="cpu", tol=1e-5, time_limit=60)
        assert r2.status == 2, r2.termination
        # certified upper bound must be >= true max
        assert r2.objbound >= -hmax["objective"] - 1e-6

    def test_objective_var_entry_is_feasible_primal(self):
        _, _, fake, out_var = make_net_lp()
        r = _BACKEND.solve_gurobi_model_with_mlxpdlp(
            fake, objective_var=out_var, sense="min",
            device="cpu", tol=1e-5, time_limit=60)
        assert r.status == 2
        # x[obj] is a primal value; the certified dual bound stays separate.
        assert r.solcount in (0, 1)
        if r.solcount:
            assert r.objval is not None
            assert r.x[out_var.VarName] == pytest.approx(r.objval)

    def test_max_primal_objective_keeps_original_sign(self):
        lp = _BACKEND.LPProblem(
            num_variables=1, num_constraints=0,
            row_ptr=np.array([0], dtype=np.int64),
            col_indices=np.array([], dtype=np.int64),
            values=np.array([], dtype=np.float64),
            variable_lower_bounds=np.array([-1.0]),
            variable_upper_bounds=np.array([2.0]),
            constraint_lower_bounds=np.array([], dtype=np.float64),
            constraint_upper_bounds=np.array([], dtype=np.float64),
            objective=np.array([0.0]), variable_names=["x"],
        )
        fake = FakeGurobiModel(lp)
        x = fake.getVarByName("x")
        r = _BACKEND.solve_gurobi_model_with_mlxpdlp(
            fake, objective_var=x, sense="max", device="cpu", tol=1e-6,
            time_limit=10, fallback=[])
        assert r.solcount == 1
        assert r.objval == pytest.approx(2.0, abs=1e-5)
        assert r.x["x"] == pytest.approx(2.0, abs=1e-5)
        assert r.objbound >= 2.0 - 1e-5

    def test_model_time_limit_caps_entire_backend_ladder(self, monkeypatch):
        _, _, fake, out_var = make_net_lp()
        fake.Params = types.SimpleNamespace(TimeLimit=0.25)
        captured = {}

        def fake_certified(lp, **kwargs):
            captured["time_limit"] = kwargs["time_limit"]
            return _BACKEND.CertifiedSolve(
                status=9, objective=None, bound=None, bound_ok=False,
                audit_gap=1.0, stationarity_abs=1.0, primal_viol=1.0,
                time_sec=0.0, num_iter=0, device="fake",
                termination="TIME_LIMIT")

        monkeypatch.setattr(_BACKEND, "solve_certified", fake_certified)
        r = _BACKEND.solve_gurobi_model_with_mlxpdlp(
            fake, objective_var=out_var, sense="min", time_limit=10.0,
            fallback=[])
        assert captured["time_limit"] == pytest.approx(0.25)
        assert r.objbound == -np.inf

    def test_model_owned_max_objective_restores_sense_and_constant(self):
        lp = _BACKEND.LPProblem(
            num_variables=1, num_constraints=0,
            row_ptr=np.array([0], dtype=np.int64),
            col_indices=np.array([], dtype=np.int64),
            values=np.array([], dtype=np.float64),
            variable_lower_bounds=np.array([0.0]),
            variable_upper_bounds=np.array([1.0]),
            constraint_lower_bounds=np.array([], dtype=np.float64),
            constraint_upper_bounds=np.array([], dtype=np.float64),
            objective=np.array([2.0]), objective_constant=5.0,
            variable_names=["x"],
        )
        fake = FakeGurobiModel(lp)
        fake.ModelSense = -1
        fake.ObjCon = 5.0
        r = _BACKEND.solve_gurobi_model_with_mlxpdlp(
            fake, objective_var=None, device="cpu", tol=1e-6,
            time_limit=10.0, fallback=[])
        assert r.objval == pytest.approx(7.0, abs=1e-5)
        assert r.objbound >= 7.0 - 1e-5
        assert r.x["x"] == pytest.approx(1.0, abs=1e-5)


class TestBuildModelPrimals:
    def test_backend_get_primals_uses_returned_solution_map(self, monkeypatch):
        lp = _BACKEND.LPProblem(
            num_variables=2, num_constraints=0,
            row_ptr=np.array([0], dtype=np.int64),
            col_indices=np.array([], dtype=np.int64),
            values=np.array([], dtype=np.float64),
            variable_lower_bounds=np.array([0.0, -1.0]),
            variable_upper_bounds=np.array([1.0, 1.0]),
            constraint_lower_bounds=np.array([], dtype=np.float64),
            constraint_upper_bounds=np.array([], dtype=np.float64),
            objective=np.zeros(2), variable_names=["input", "out"],
        )
        model = FakeGurobiModel(lp)
        input_var = model.getVarByName("input")
        out_var = model.getVarByName("out")

        class Node:
            def __init__(self, solver_vars, inputs=None):
                self.solver_vars = solver_vars
                self.inputs = [] if inputs is None else inputs

        input_node = Node([input_var])
        final_node = Node([out_var], [input_node])

        class Net:
            final_name = "final"
            relus = []
            solver_model = model
            def final_node(self):
                return final_node
            def __getitem__(self, name):
                assert name == self.final_name
                return final_node

        class ModelOri:
            @staticmethod
            def children():
                return []

        class Wrapper:
            net = Net()
            model_ori = ModelOri()
            def build_solver_model(self, *args, **kwargs):
                return None

        result = GurobiLikeResult(
            status=2, objval=0.25, objbound=0.2, solcount=1,
            x={"input": 0.75, "out": 0.25}, solve_time=0.0,
            device="cpu", certified=True, termination="OPTIMAL")
        monkeypatch.setattr(bounds_core, "mlxpdlp_enabled", lambda: True)
        monkeypatch.setattr(bounds_core, "get_lp_backend_settings",
                            lambda: dict(SETTINGS_CPU))
        monkeypatch.setattr(bounds_core, "make_fallback_from_settings",
                            lambda s: s["fallback"])
        monkeypatch.setattr(
            bounds_core, "solve_gurobi_model_with_mlxpdlp",
            lambda *a, **k: result)
        glbs = bounds_core.build_the_model_lp(
            Wrapper(), using_integer=False, get_primals=True)
        assert glbs == [0.2]


class TestAllNodeSplitLP:
    def _run(self, monkeypatch, fake, out_var_name, rhs,
             result=None):
        monkeypatch.setattr(bounds_core, "mlxpdlp_enabled", lambda: True)
        monkeypatch.setattr(bounds_core, "get_lp_backend_settings",
                            lambda: dict(SETTINGS_CPU))
        monkeypatch.setattr(bounds_core, "make_fallback_from_settings",
                            lambda s: s["fallback"])
        monkeypatch.setattr(bounds_core, "copy_model",
                            lambda model, **kw: model)
        if result is not None:
            monkeypatch.setattr(bounds_core,
                                "solve_gurobi_model_with_mlxpdlp", result)
        bounds_core.multiprocess_lp_model = fake
        bounds_core.input_name = [f"var_{i}" for i in range(8)]
        bounds_core.termination_flag_lp = types.SimpleNamespace(value=0)

        class _Stub:
            def size(self, dim=0):
                return 1
        # rhs must be indexable with len == len(orig_out_vars) (production
        # passes a 1-D tensor of thresholds).
        arg = ([], [], [out_var_name], [_Stub()], [_Stub()], [rhs], 0)
        return bounds_core.all_node_split_LP(arg)

    def test_safe_when_certified_bound_above_threshold(self, monkeypatch):
        lp, info, fake, out_var = make_net_lp()
        h = _PROBE.solve_highs(lp)
        opt = h["objective"]
        lp_status, dix, glb, cex = self._run(
            monkeypatch, fake, out_var.VarName, opt - 1.0)
        assert lp_status == "safe"
        assert glb > opt - 1.0
        assert cex is None

    def test_unsafe_with_feasible_counterexample(self, monkeypatch):
        lp, info, fake, out_var = make_net_lp()
        h = _PROBE.solve_highs(lp)
        opt = h["objective"]
        lp_status, dix, glb, cex = self._run(
            monkeypatch, fake, out_var.VarName, opt + 1.0)
        assert lp_status == "unsafe"
        assert cex is not None and len(cex) == 8
        assert all(isinstance(v, float) for v in cex)

    def test_unknown_when_backend_undecided(self, monkeypatch):
        lp, info, fake, out_var = make_net_lp()

        def fake_solve(*a, **k):
            return GurobiLikeResult(status=9, objval=None, objbound=-np.inf,
                                    solcount=0, x={}, solve_time=0.1,
                                    device="cpu", certified=False,
                                    termination="TIME_LIMIT")
        lp_status, dix, glb, cex = self._run(
            monkeypatch, fake, out_var.VarName, 0.0, result=fake_solve)
        assert lp_status == "unknown"
        assert cex is None


class TestLpSolverSite:
    def test_lp_solver_backend_refines_soundly(self, monkeypatch):
        import lp_mip_solver.utils as su
        lp, info, fake, out_var = make_net_lp()
        # pick an unstable hidden neuron: force bounds straddling zero
        y_lo = info["y_ranges"][0][0]
        v = fake.getVarByName(f"var_{y_lo}")
        v.LB, v.UB = -0.5, 0.5
        monkeypatch.setattr(bounds_core, "mlxpdlp_enabled", lambda: True)
        monkeypatch.setattr(bounds_core, "get_lp_backend_settings",
                            lambda: dict(SETTINGS_CPU))
        monkeypatch.setattr(bounds_core, "make_fallback_from_settings",
                            lambda s: s["fallback"])
        monkeypatch.setattr(su, "stop_multiprocess", False)
        bounds_core.multiprocess_lp_model = fake
        vlb, vub, print_str, refined = bounds_core.lp_solver(v.VarName)
        assert refined
        # Certified bounds stay sound w.r.t. the box: -0.5 <= vlb, vub <= 0.5
        assert vlb >= -0.5 - 1e-6 and vlb <= vub
        assert vub <= 0.5 + 1e-6
        assert "mlxPDLP" in print_str


class TestMipSolverLbUb:
    def _patch(self, monkeypatch, fake):
        monkeypatch.setattr(solver_utils, "mlxpdlp_enabled", lambda: True)
        monkeypatch.setattr(solver_utils, "get_lp_backend_settings",
                            lambda: dict(SETTINGS_CPU))
        monkeypatch.setattr(solver_utils, "make_fallback_from_settings",
                            lambda s: s["fallback"])
        fake_args = types.ModuleType("arguments")
        fake_args.Config = {
            "solver": {"mip": {"early_stop": False, "mip_solver": "gurobi"}},
            "bab": {"timeout": 3600.0},
        }
        # The worker reads the module reference it imported, not sys.modules.
        monkeypatch.setattr(solver_utils, "arguments", fake_args)
        solver_utils.multiprocess_mip_model = fake
        solver_utils.stop_multiprocess = False
        solver_utils.mip_solve_time_start = time.time()

    def test_mip_solver_lb_ub_backend(self, monkeypatch):
        lp, info, fake, out_var = make_net_lp()
        self._patch(monkeypatch, fake)
        vlb, vub, status, adv = solver_utils.mip_solver_lb_ub(
            out_var.VarName)
        assert status == 2, status
        assert vlb <= vub
        assert np.isfinite(vlb) and np.isfinite(vub)

    def test_mip_solver_lb_ub_and_backend_feasible(self, monkeypatch):
        lp, info, fake, out_var = make_net_lp()
        h = _PROBE.solve_highs(lp)
        opt = h["objective"]
        self._patch(monkeypatch, fake)
        _, _, status, adv = solver_utils.mip_solver_lb_ub_and(
            [out_var.VarName], rhs=[opt + 1.0])
        assert status == 2

    def test_mip_solver_lb_ub_and_backend_infeasible(self, monkeypatch):
        lp, info, fake, out_var = make_net_lp()
        h = _PROBE.solve_highs(lp)
        opt = h["objective"]
        self._patch(monkeypatch, fake)
        _, _, status, adv = solver_utils.mip_solver_lb_ub_and(
            [out_var.VarName], rhs=[opt - 1.0])
        assert status == 3

    def test_and_mode_never_reports_feasible_without_primal(self, monkeypatch):
        _, _, fake, out_var = make_net_lp()
        self._patch(monkeypatch, fake)

        def no_primal(*args, **kwargs):
            return GurobiLikeResult(
                status=2, objval=None, objbound=0.0, solcount=0, x={},
                solve_time=0.0, device="cpu", certified=True,
                termination="OPTIMAL", primal_viol=1.0)

        monkeypatch.setattr(
            solver_utils, "solve_gurobi_model_with_mlxpdlp", no_primal)
        _, _, status, adv = solver_utils.mip_solver_lb_ub_and(
            [out_var.VarName], rhs=[0.0])
        assert status == _BACKEND.STATUS_TIME_LIMIT
        assert adv is None

    def test_backend_call_uses_remaining_shared_timeout(self, monkeypatch):
        _, _, fake, out_var = make_net_lp()
        self._patch(monkeypatch, fake)
        solver_utils.arguments.Config["bab"]["timeout"] = 1.0
        solver_utils.mip_solve_time_start = time.time() - 0.8
        captured = {}

        def capture(*args, **kwargs):
            captured["time_limit"] = kwargs["time_limit"]
            return GurobiLikeResult(
                status=9, objval=None, objbound=-np.inf, solcount=0, x={},
                solve_time=0.0, device="cpu", certified=False,
                termination="TIME_LIMIT")

        monkeypatch.setattr(
            solver_utils, "solve_gurobi_model_with_mlxpdlp", capture)
        solver_utils.mip_solver_lb_ub(out_var.VarName)
        assert 0.0 < captured["time_limit"] < 0.3

    def test_feasible_early_stop_uses_witness_status(self, monkeypatch):
        _, _, fake, out_var = make_net_lp()
        out_var.LB, out_var.UB = -1.0, 1.0
        self._patch(monkeypatch, fake)

        def early_witness(*args, **kwargs):
            return GurobiLikeResult(
                status=_BACKEND.STATUS_TIME_LIMIT,
                objval=-0.25, objbound=-0.5, solcount=1,
                x={v.VarName: 0.0 for v in fake.getVars()},
                solve_time=0.0, device="cpu", certified=True,
                termination="TIME_LIMIT")

        monkeypatch.setattr(
            solver_utils, "solve_gurobi_model_with_mlxpdlp", early_witness)
        _, vub, status, _ = solver_utils.mip_solver_lb_ub(out_var.VarName)
        assert vub == pytest.approx(-0.25)
        assert status == 15  # Gurobi USER_OBJ_LIMIT-compatible witness status


class TestSettings:
    def test_defaults_without_arguments_module(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "arguments", raising=False)
        s = _BACKEND.get_lp_backend_settings()
        assert s["backend"] == "gurobi"
        assert _BACKEND.mlxpdlp_enabled() is False

    def test_config_override_and_normalization(self, monkeypatch):
        fake_args = types.ModuleType("arguments")
        fake_args.Config = {
            "solver": {"mip": {
                "lp_backend": "mlxpdlp", "mlxpdlp_device": "metal",
                "mlxpdlp_tolerance": 1e-4, "mlxpdlp_margin": 1e-2,
                "mlxpdlp_fallback": "cpu"}},
            "bab": {"timeout": 100.0},
        }
        monkeypatch.setitem(sys.modules, "arguments", fake_args)
        s = _BACKEND.get_lp_backend_settings()
        assert s["backend"] == "mlxpdlp"
        assert s["device"] == "gpu"  # metal normalized
        assert s["tol"] == 1e-4
        assert s["margin"] == 1e-2
        assert s["fallback"] == ["cpu"]
        assert s["time_limit"] == 100.0
        assert _BACKEND.mlxpdlp_enabled() is True
