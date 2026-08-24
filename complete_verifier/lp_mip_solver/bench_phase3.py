#!/usr/bin/env python3
"""Phase 3 benchmark: Metal vs CPU vs HiGHS/Gurobi crossover + PDLP tuning.

Driver-side wall-clock timing (mlxPDLP's cumulative_time_sec was observed
to over-report inside verifier processes; see plan doc section 13).

Run:
  python complete_verifier/lp_mip_solver/bench_phase3.py --quick
  python complete_verifier/lp_mip_solver/bench_phase3.py \
      --sizes small,medium,large --grid --conditioning
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "complete_verifier"))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

probe = _load("probe_p3", os.path.join(_REPO, "complete_verifier",
                                       "lp_mip_solver", "probe_mlxpdlp.py"))
mb = _load("mb_p3", os.path.join(_REPO, "complete_verifier", "lp_mip_solver",
                                 "mlxpdlp_backend.py"))

RESULTS = []


def save_results(path="/tmp/phase3_bench.json"):
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def solve_cfg(lp, device, tol, time_limit, sense="min"):
    return timed(lambda: mb.solve_certified(
        lp, device=device, tol=tol, time_limit=time_limit, sense=sense,
        fallback=[]))


def solve_ladder(lp, device, tol, time_limit, margin, threshold, fallback):
    return timed(lambda: mb.solve_certified(
        lp, device=device, tol=tol, time_limit=time_limit, sense="min",
        margin=margin, threshold=threshold, fallback=fallback))


def sweep_size(name, sizes, seed, time_limit, objective_indices):
    lp_full, info = probe.build_relu_mlp_lp(sizes, seed=seed)
    y_out = info["y_out_range"]
    objs = {"out0": y_out[0]}
    for i, oi in enumerate(objective_indices):
        objs[f"out{oi}"] = y_out[0] + oi
    for oname, oidx in objs.items():
        lp = lp_full.copy()
        print(f"[{name}/{oname}] start", flush=True)
        save_results()
        lp.objective = np.zeros(lp.num_variables)
        lp.objective[oidx] = 1.0
        # reference: HiGHS (capped)
        try:
            h, ht = timed(lambda: probe.solve_highs(
                lp, time_limit=300 if lp.nnz > 3_000_000 else None))
            ref = h["objective"] if h["success"] else None
        except Exception as e:
            h, ht, ref = {"success": False}, 0.0, None
        RESULTS.append(dict(kind="solve", size=name, obj=oname,
                            solver="highs", time=ht, status=str(h.get("status")),
                            obj_val=ref, wall="driver"))
        save_results()
        print(f"[{name}/{oname}] highs {ht:.1f}s", flush=True)
        for dev, tol, tl in (("gpu", 1e-4, time_limit),
                             ("cpu", 1e-5, time_limit),
                             ("cpu", 1e-6, time_limit)):
            cs, wt = solve_cfg(lp, dev, tol, tl)
            RESULTS.append(dict(kind="solve", size=name, obj=oname,
                                solver=f"{dev}@{tol:.0e}", time=wt,
                                status=cs.status, bound=cs.bound,
                                bound_ok=cs.bound_ok, audit=cs.audit_gap,
                                term=cs.termination, wall="driver"))
            save_results()
            print(f"[{name}/{oname}] {dev}@{tol:.0e} {wt:.1f}s "
                  f"{cs.termination} audit={cs.audit_gap:.1e}", flush=True)
        # gurobi (restricted license; tiny/small only)
        if lp.num_variables <= 2000:
            try:
                cs, wt = timed(lambda: mb.solve_gurobi_lp(lp, time_limit=60))
                RESULTS.append(dict(kind="solve", size=name, obj=oname,
                                    solver="gurobi", time=wt, status=cs.status,
                                    obj_val=cs.objective, wall="driver"))
            except Exception as e:
                RESULTS.append(dict(kind="solve", size=name, obj=oname,
                                    solver="gurobi", time=0.0, status="err",
                                    note=str(e)[:80], wall="driver"))
        # ladder: margin case that FORCES escalation to the certified stage
        if ref is not None:
            cs, wt = solve_ladder(lp, "gpu", 1e-4, time_limit, margin=1e-3,
                                  threshold=ref + abs(ref) * 0.01 + 1e-2,
                                  fallback=["cpu"])
            RESULTS.append(dict(kind="ladder", size=name, obj=oname,
                                time=wt, status=cs.status, device=cs.device,
                                bound=cs.bound, audit=cs.audit_gap,
                                wall="driver"))
    print(f"size {name}: done", flush=True)


def run_grid(name, sizes, seed, time_limit):
    """PDLP parameter grid on one medium objective."""
    lp, info = probe.build_relu_mlp_lp(sizes, seed=seed)
    lp.objective = np.zeros(lp.num_variables)
    lp.objective[info["y_out_range"][0]] = 1.0
    for restart_policy in (0, 1, 2):
        for ruiz in (0, 10):
            for curt in (0, 20):
                params = mb.make_parameters(tol=1e-5, time_limit=time_limit)
                params.restart_policy = restart_policy
                params.l_inf_ruiz_iterations = ruiz
                params.curtis_reid_iterations = curt
                def fn(p=params):
                    return mb.solve_mlxpdlp(lp, device="cpu", tol=1e-5,
                                            time_limit=time_limit, parameters=p)
                out, wt = timed(fn)
                RESULTS.append(dict(kind="grid", size=name,
                                    restart_policy=restart_policy, ruiz=ruiz,
                                    curtis_reid=curt, time=wt,
                                    status=out["status"],
                                    iters=out["num_iter"],
                                    audit=out["audit_gap"], wall="driver"))
                print(f"grid rp={restart_policy} ruiz={ruiz} cr={curt}: "
                      f"{wt:.1f}s {out['num_iter']}it {out['status']}", flush=True)


def run_conditioning(name, sizes, seed, time_limit):
    """Objective-scale sensitivity: scale the final layer outputs."""
    lp, info = probe.build_relu_mlp_lp(sizes, seed=seed)
    y0 = info["y_out_range"][0]
    for scale in (1.0, 10.0, 100.0):
        lp2 = lp.copy()
        # scale objective only (mimics margin scale variation)
        lp2.objective = np.zeros(lp2.num_variables)
        lp2.objective[y0] = scale
        for dev, tol in (("gpu", 1e-4), ("cpu", 1e-5)):
            cs, wt = solve_cfg(lp2, dev, tol, time_limit)
            RESULTS.append(dict(kind="conditioning", size=name, scale=scale,
                                solver=f"{dev}@{tol:.0e}", time=wt,
                                status=cs.status, audit=cs.audit_gap,
                                term=cs.termination, wall="driver"))
            print(f"cond scale={scale} {dev}: {wt:.1f}s {cs.termination}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="small,medium",
                    help="comma list of tiny,small,medium,large,xlarge")
    ap.add_argument("--quick", action="store_true",
                    help="one objective per size, short time limits")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--conditioning", action="store_true")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--out-json", default="/tmp/phase3_bench.json")
    args = ap.parse_args()

    for name in args.sizes.split(","):
        name = name.strip()
        sizes = probe.SIZES[name]
        objs = [0] if args.quick else [0, 2, 5]
        sweep_size(name, sizes, seed=0, time_limit=args.time_limit,
                   objective_indices=objs)
    if args.grid:
        run_grid("medium", probe.SIZES["medium"], seed=0,
                 time_limit=args.time_limit)
    if args.conditioning:
        run_conditioning("medium", probe.SIZES["medium"], seed=0,
                         time_limit=args.time_limit)

    with open(args.out_json, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"results -> {args.out_json} ({len(RESULTS)} records)")


if __name__ == "__main__":
    main()
