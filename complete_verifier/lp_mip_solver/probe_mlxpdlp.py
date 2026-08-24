#!/usr/bin/env python3
"""Phase 0 probe: is mlxPDLP (Metal FP32 / CPU FP64) competitive for
alpha-beta-CROWN-style network LP relaxations?

Builds the LP relaxation of a ReLU MLP directly as CSR (the same problem
shape the repo's `build_the_model_lp` / `all_node_split_LP` solve), then:
  - baseline: scipy.optimize.linprog (HiGHS, exact simplex/IPM reference)
  - mlxPDLP device="cpu"   (float64, Accelerate SpMV)
  - mlxPDLP device="gpu"   (float32 Metal SpMV; ~1e-4 practical tolerance)
and audits every mlxPDLP result with a host-float64 stationarity check on
the original (unscaled) model via mlxpdlp_backend.audit_certificate.

Usage:
    python probe_mlxpdlp.py [--sizes small,medium,large,xlarge]
                            [--skip-highs] [--warm-start] [--seed 0]
Run from the repo root with an environment that has numpy, scipy, mlxpdlp:
    uv run --python /path/to/probe-venv python complete_verifier/lp_mip_solver/probe_mlxpdlp.py
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

# Load mlxpdlp_backend.py directly by file path: importing it through the
# complete_verifier.lp_mip_solver package would drag in torch/gurobipy, which
# are not needed for this probe. Fall back to the normal package import when
# the repo environment (with torch) is available.
_BACKEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "mlxpdlp_backend.py")
try:
    _spec = importlib.util.spec_from_file_location("mlxpdlp_backend", _BACKEND_PATH)
    _backend = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_backend)
    LPProblem = _backend.LPProblem
    solve_mlxpdlp = _backend.solve_mlxpdlp
    audit_certificate = _backend.audit_certificate
    make_parameters = _backend.make_parameters
except Exception:
    from complete_verifier.lp_mip_solver.mlxpdlp_backend import (  # noqa: E402
        LPProblem, solve_mlxpdlp, audit_certificate, make_parameters)


# ---------------------------------------------------------------------------
# Network LP relaxation builder (CSR, no solver dependency)
# ---------------------------------------------------------------------------

SIZES = {
    "tiny":   (16, 32, 32, 4),
    "small":  (100, 256, 256, 10),
    "medium": (784, 1024, 1024, 10),
    "large":  (784, 2048, 2048, 10),
    "xlarge": (3072, 4096, 4096, 10),
}


def build_relu_mlp_lp(sizes, seed=0, output_index=0, eps=1e-8):
    """LP relaxation of MLP ReLU network with box input [-1, 1]^n0.

    Variables (order): x (n0), then per hidden layer i: y_i (h_i), z_i (h_i),
    and finally y_out (n_out).

    Constraints:
      eq:      y_i - W_i z_{i-1} = b_i          (z_{-1} := x)
      ReLU triangle (lb<0<ub):  z >= y, z >= 0, z <= ub/(ub-lb)*(y-lb)
      stable:  z = 0 (ub<=0) or z = y (lb>=0)

    Returns (lp, info) where info has the index ranges of every block and
    the IBP pre-activation bounds used for the triangles.
    """
    rng = np.random.default_rng(seed)
    n_in = sizes[0]
    hidden = sizes[1:-1]
    n_out = sizes[-1]

    # Weights: He-ish init, biases small.
    layers = []
    prev = n_in
    for h in list(hidden) + [n_out]:
        W = rng.standard_normal((h, prev)) * np.sqrt(2.0 / prev)
        b = rng.standard_normal(h) * 0.1
        layers.append((W, b))
        prev = h

    # Variable layout.
    n_vars = n_in + 2 * sum(hidden) + n_out
    x_range = (0, n_in)
    y_ranges, z_ranges = [], []
    off = n_in
    for h in hidden:
        y_ranges.append((off, off + h))
        off += h
        z_ranges.append((off, off + h))
        off += h
    y_out_range = (off, off + n_out)

    # Interval-bound propagation (IBP) for pre-activation bounds.
    lb = -np.ones(n_in)
    ub = np.ones(n_in)
    pre_bounds = []  # (lb, ub) for each hidden layer pre-activation
    for (W, b) in layers[:-1]:
        Wp = np.clip(W, 0, None)
        Wn = np.clip(W, None, 0)
        lb_new = Wp @ lb + Wn @ ub + b
        ub_new = Wp @ ub + Wn @ lb + b
        pre_bounds.append((lb_new, ub_new))
        # Pass ReLU bounds forward.
        zlb = np.clip(lb_new, 0, None)
        zub = np.clip(ub_new, 0, None)
        lb, ub = zlb, zub

    # Sparse matrix construction.
    rows, cols, vals = [], [], []
    con_lb, con_ub = [], []
    n_cons = 0

    def add_row(inds, coeffs, l, u):
        nonlocal n_cons
        rows.append(np.asarray(inds, dtype=np.int64))
        cols.append(np.asarray(inds, dtype=np.int64))
        vals.append(np.asarray(coeffs, dtype=np.float64))
        con_lb.append(l)
        con_ub.append(u)
        n_cons += 1

    # Layer equations.
    z_prev_range = x_range
    for li, (W, b) in enumerate(layers):
        y_lo, y_hi = y_ranges[li] if li < len(hidden) else y_out_range
        for i in range(W.shape[0]):
            row_inds = list(z_prev_range[0] + np.arange(z_prev_range[1] - z_prev_range[0])) + [y_lo + i]
            row_coeffs = list(-W[i]) + [1.0]
            # y_i - W z_{i-1} = b_i
            add_row(row_inds, row_coeffs, float(b[i]), float(b[i]))
        if li < len(hidden):
            z_prev_range = z_ranges[li]

    # ReLU triangles / fixing.
    for li, (lbl, ubl) in enumerate(pre_bounds):
        y_lo, y_hi = y_ranges[li]
        z_lo, z_hi = z_ranges[li]
        unstable = (lbl < -eps) & (ubl > eps)
        for i in range(y_hi - y_lo):
            yi, zi = y_lo + i, z_lo + i
            if not unstable[i]:
                # Fix z = 0 (ubl <= 0) or z = y (lbl >= 0) with an equality row.
                if ubl[i] <= eps:
                    add_row([zi], [1.0], 0.0, 0.0)
                else:
                    add_row([yi, zi], [-1.0, 1.0], 0.0, 0.0)
                continue
            # z - y >= 0
            add_row([zi, yi], [1.0, -1.0], 0.0, np.inf)
            # z <= slope * (y - lb)  <=>  slope*y - z >= slope*lb
            slope = ubl[i] / (ubl[i] - lbl[i])
            add_row([yi, zi], [slope, -1.0], slope * lbl[i], np.inf)

    con_lb = np.array(con_lb, dtype=np.float64)
    con_ub = np.array(con_ub, dtype=np.float64)
    row_ptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    col_indices = np.concatenate([c for c in cols]).astype(np.int64) if rows else np.array([], dtype=np.int64)
    values = np.concatenate([v for v in vals]).astype(np.float64) if vals else np.array([], dtype=np.float64)

    # Final (output) layer IBP bounds: in the real verifier these come
    # from the LiRPA backward pass. They do not change the LP optimum
    # (outer bounds) but make the rigorous certificate correction finite.
    W_out, b_out = layers[-1]
    Wp_out = np.clip(W_out, 0, None)
    Wn_out = np.clip(W_out, None, 0)
    lb_out = Wp_out @ lb + Wn_out @ ub + b_out
    ub_out = Wp_out @ ub + Wn_out @ lb + b_out

    # Variable bounds.
    var_lb = np.full(n_vars, -np.inf)
    var_ub = np.full(n_vars, np.inf)
    var_lb[x_range[0]:x_range[1]] = -1.0
    var_ub[x_range[0]:x_range[1]] = 1.0
    var_lb[y_out_range[0]:y_out_range[1]] = lb_out
    var_ub[y_out_range[0]:y_out_range[1]] = ub_out
    for li, (lbl, ubl) in enumerate(pre_bounds):
        y_lo, y_hi = y_ranges[li]
        z_lo, z_hi = z_ranges[li]
        unstable = (lbl < -eps) & (ubl > eps)
        # z in [max(lb,0), max(ub,0)]; stable z pinned exactly.
        zl = np.clip(lbl, 0, None)
        zu = np.clip(ubl, 0, None)
        zl[~unstable] = np.where(ubl[~unstable] <= eps, 0.0, lbl[~unstable])
        zu[~unstable] = np.where(ubl[~unstable] <= eps, 0.0, ubl[~unstable])
        var_lb[y_lo:y_hi] = lbl
        var_ub[y_lo:y_hi] = ubl
        var_lb[z_lo:z_hi] = zl
        var_ub[z_lo:z_hi] = zu

    # Objective: minimize y_out[output_index].
    objective = np.zeros(n_vars)
    objective[y_out_range[0] + output_index] = 1.0

    lp = LPProblem(
        num_variables=n_vars, num_constraints=n_cons,
        row_ptr=row_ptr, col_indices=col_indices, values=values,
        objective=objective, objective_constant=0.0,
        variable_lower_bounds=var_lb, variable_upper_bounds=var_ub,
        constraint_lower_bounds=con_lb, constraint_upper_bounds=con_ub,
    )
    info = {
        "x_range": x_range, "y_ranges": y_ranges, "z_ranges": z_ranges,
        "y_out_range": y_out_range, "pre_bounds": pre_bounds,
        "n_unstable": [int(((lbl < -eps) & (ubl > eps)).sum())
                       for lbl, ubl in pre_bounds],
    }
    return lp, info


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def solve_highs(lp, time_limit=None, output_index=None):
    """Exact reference solve via scipy HiGHS (dual simplex/IPM).

    Rows are split: con_lb == con_ub -> A_eq; con_ub == inf -> -A >= -con_lb
    (i.e. A_ub with negated row); con_lb == -inf -> A_ub; two-sided finite ->
    both an upper and a lower (negated) row.
    """
    from scipy.optimize import linprog
    A = lp.scipy_csr()
    clb = lp.constraint_lower_bounds
    cub = lp.constraint_upper_bounds

    eq = clb == cub
    up_only = (clb == -np.inf) & (cub != np.inf)
    lo_only = (clb != -np.inf) & (cub == np.inf)
    two_sided = ~(eq | up_only | lo_only)

    A_eq, b_eq, A_ub, b_ub = None, None, None, None
    if eq.any():
        A_eq = A[eq]
        b_eq = cub[eq]
    if up_only.any():
        A_ub = A[up_only]
        b_ub = cub[up_only]
    if lo_only.any():
        A_ub = scipy_vstack_neg(A_ub, -A[lo_only])
        b_ub = np.concatenate([b_ub, -clb[lo_only]]) if b_ub is not None else -clb[lo_only]
    if two_sided.any():
        A_ub = scipy_vstack_neg(A_ub, A[two_sided])
        b_ub = np.concatenate([b_ub, cub[two_sided]]) if b_ub is not None else cub[two_sided]
        A_ub = scipy_vstack_neg(A_ub, -A[two_sided])
        b_ub = np.concatenate([b_ub, -clb[two_sided]]) if b_ub is not None else -clb[two_sided]

    t0 = time.perf_counter()
    res = linprog(
        lp.objective,
        A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        bounds=list(zip(lp.variable_lower_bounds, lp.variable_upper_bounds)),
        method="highs",
        options={"time_limit": time_limit} if time_limit else None,
    )
    return {
        "time_sec": time.perf_counter() - t0,
        "status": res.status, "message": res.message,
        "objective": float(res.fun) if res.fun is not None else None,
        "success": bool(res.success),
    }


def scipy_vstack_neg(base, block):
    """Stack a block onto an existing csr matrix (or start one)."""
    import scipy.sparse as sp
    if base is None:
        return block.copy()
    return sp.vstack([base, block], format="csr")


def objective_on_variable(lp, var_index, maximize=False):
    """Return a copy of the LP minimizing (+1) or maximizing (-1) one variable.
    For a maximize objective the reported objective must be sign-flipped
    (mlxPDLP always minimizes)."""
    lp2 = lp.copy()
    lp2.objective = np.zeros(lp.num_variables)
    lp2.objective[var_index] = -1.0 if maximize else 1.0
    return lp2


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_case(name, lp, device, tol, presolve=True, warm_start=None,
             time_limit=3600.0, iter_limit=0, host_polish=False):
    parameters = None
    if host_polish or iter_limit > 0:
        parameters = make_parameters(tol=tol, time_limit=time_limit,
                                     presolve=presolve,
                                     iteration_limit=iter_limit)
        if host_polish:
            parameters.host_double_polishing = True
            parameters.host_double_early_handoff = True
    out = solve_mlxpdlp(lp, device=device, tol=tol, presolve=presolve,
                        time_limit=time_limit, warm_start=warm_start,
                        parameters=parameters)
    res = out["result"]
    out["case"] = name
    out["device"] = device
    out["tol"] = tol
    out["nnz"] = lp.nnz
    out["nvars"] = lp.num_variables
    out["ncons"] = lp.num_constraints
    out["preprocess_sec"] = float(res.rescaling_time_sec + res.presolve_time)
    return out


def fmt(x, nd=4):
    return "None" if x is None else f"{x:.{nd}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="tiny,small,medium,large",
                    help="comma list of " + ",".join(SIZES))
    ap.add_argument("--skip-highs", action="store_true")
    ap.add_argument("--warm-start", action="store_true",
                    help="also run the per-neuron warm-start pattern on CPU FP64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--cpu-tol", type=float, default=1e-7)
    ap.add_argument("--metal-tol", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=3600.0,
                    help="per-solve time limit in seconds (mlxPDLP); "
                         "TIME_LIMIT status is recorded, not fatal")
    ap.add_argument("--metal-polish", action="store_true",
                    help="enable mlxPDLP host-FP64 polishing on Metal runs "
                         "(checks whether Metal can pass the 1e-6 cert audit)")
    ap.add_argument("--iter-limit", type=int, default=0,
                    help="additional per-solve iteration cap (0 = unlimited)")
    args = ap.parse_args()

    import mlxpdlp
    print(f"mlxPDLP {mlxpdlp.version()}  has_gpu={mlxpdlp.has_gpu()}")
    print(f"CPU tol {args.cpu_tol} (FP64), Metal tol {args.metal_tol} (FP32 floor ~1e-4)")

    results = []

    def save_progress():
        if args.out_json:
            clean = []
            for r in results:
                d = {}
                for k, v in r.items():
                    if k == "result":  # SolveResult is not JSON-serializable
                        continue
                    try:
                        json.dumps(v)
                        d[k] = v
                    except TypeError:
                        continue
                clean.append(d)
            with open(args.out_json, "w") as f:
                json.dump(clean, f, indent=2)

    for name in args.sizes.split(","):
        name = name.strip()
        if name not in SIZES:
            print(f"skip unknown size {name}")
            continue
        sizes = SIZES[name]
        lp, info = build_relu_mlp_lp(sizes, seed=args.seed)
        print(f"\n=== {name} {sizes}: nvars={lp.num_variables} "
              f"ncons={lp.num_constraints} nnz={lp.nnz} "
              f"unstable={info['n_unstable']} ===")

        if not args.skip_highs:
            try:
                if lp.nnz > 10_000_000:
                    h = {"objective": None, "time_sec": 0.0, "status": "skipped",
                         "message": "model too large for the HiGHS baseline"}
                    print("  HiGHS: skipped (nnz > 10M)")
                else:
                    h = solve_highs(lp, time_limit=120 if lp.nnz > 3_000_000 else None)
                    print(f"  HiGHS: obj={fmt(h['objective'])} time={h['time_sec']:.2f}s "
                          f"status={h['status']} ({h['message']})")
            except Exception as e:
                h = {"objective": None, "time_sec": 0.0, "status": "error", "message": str(e)}
                print(f"  HiGHS failed: {e}")
        else:
            h = {"objective": None, "time_sec": 0.0, "status": "skipped", "message": ""}

        for device, tol in (("cpu", args.cpu_tol), ("gpu", args.metal_tol)):
            try:
                r = run_case(name, lp, device, tol, time_limit=args.time_limit,
                             iter_limit=args.iter_limit,
                             host_polish=(args.metal_polish and device == "gpu"))
                gap = (None if h["objective"] is None or r["objective"] is None
                       else abs(r["objective"] - h["objective"]) / max(1.0, abs(h["objective"])))
                print(f"  mlxPDLP/{device:>3} tol={tol:.0e}: obj={fmt(r['objective'],6)} "
                      f"dual={fmt(r['dual_objective'],6)} time={r['time_sec']:.2f}s "
                      f"(pre={r['preprocess_sec']:.2f}s) "
                      f"iter={r['num_iter']} status={r['status']} "
                      f"|obj-opt|={fmt(gap,3)} cert_ok={r['cert_lb_ok']} "
                      f"audit_gap={fmt(r['audit_gap'],3)} prim_viol={fmt(r['primal_viol'],3)}")
                r["highs_obj"] = h["objective"]
                r["highs_time"] = h["time_sec"]
                r["gap_vs_highs"] = gap
                results.append(r)
            except Exception as e:
                print(f"  mlxPDLP/{device} FAILED: {type(e).__name__}: {e}")
        save_progress()

    # Per-neuron warm-start pattern (site #2 in the integration plan):
    # same matrix, many different single-variable objectives.
    if args.warm_start:
        print("\n=== per-neuron warm-start pattern (medium net, CPU FP64) ===")
        lp, info = build_relu_mlp_lp(SIZES["medium"], seed=args.seed)
        rng = np.random.default_rng(args.seed + 1)
        # Refine bounds of post-activation neurons of the LAST hidden layer:
        # the full network LP relaxation genuinely constrains these (matches
        # the repo's per-neuron `lp_solver` pattern).
        z_lo, z_hi = info["z_ranges"][-1]
        targets = [int(z_lo + rng.integers(0, z_hi - z_lo)) for _ in range(6)]
        prev = None  # warm start from previous solve
        rows = []
        for k, vi in enumerate(targets):
            lp_k = objective_on_variable(lp, vi, maximize=True)
            # presolve must be off to allow warm starts
            r = solve_mlxpdlp(lp_k, device="cpu", tol=args.cpu_tol,
                              presolve=False, warm_start=prev)
            prev = {"primal": r["result"].primal_solution,
                    "dual": r["result"].dual_solution,
                    "reduced_cost": r["result"].reduced_cost}
            res = r["result"]
            pre = float(res.rescaling_time_sec + res.presolve_time)
            print(f"  neuron {vi} (z_last[{vi-z_lo}], MAX): obj={fmt(-r['objective'],6)} "
                  f"time={r['time_sec']:.3f}s (pre={pre:.2f}s) "
                  f"iter={r['num_iter']} status={r['status']} "
                  f"cert_ok={r['cert_lb_ok']}")
            r["objective"] = -r["objective"]
            r["case"] = f"per-neuron #{k}"
            rows.append(r)
        # Cold-start comparison for the first neuron.
        r = solve_mlxpdlp(objective_on_variable(lp, targets[0], maximize=True),
                          device="cpu", tol=args.cpu_tol, presolve=True)
        print(f"  (cold, presolve on) neuron {targets[0]} (MAX): obj={fmt(-r['objective'],6)} "
              f"time={r['time_sec']:.3f}s iter={r['num_iter']} status={r['status']}")
        r["objective"] = -r["objective"]
        r["case"] = "per-neuron cold"
        rows.append(r)
        results.extend(rows)

    if args.out_json:
        save_progress()
        print(f"\nresults written to {args.out_json}")


if __name__ == "__main__":
    main()
