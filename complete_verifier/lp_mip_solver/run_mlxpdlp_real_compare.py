#!/usr/bin/env python3
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
"""Pre-Phase-3 sanity comparison: mlxPDLP Metal FP32 vs CPU FP64 on REAL
verifier runs over a conv/ResNet, in the three LP-relevant modes, with NO
tuning. Decides whether Phase 3 (benchmark + tune) is worth it.

Modes (per device metal/cpu):
  allnode     - bab + solver.beta-crown.all_node_split_LP (one LP per BaB
                subdomain, site #1). Requires domains to reach full split
                depth, so it is run on the SMALL conv net.
  mip-refine  - complete_verifier=mip with lp_solver=True /
                formulation=lp_integer (per-neuron LP relaxation
                refinement, site #3). Engages immediately on the full
                network; run on the cifar-resnet.
  bab-plain   - default bab WITHOUT LP (no-acceleration baseline).

The network has a fixed linear margin head (final output =
y_true - y_runnerup), so the specification is a single scalar clause
c=[[1]], rhs=0.

Usage (probe venv with torch+mlxpdlp+gurobipy):
  python complete_verifier/lp_mip_solver/run_mlxpdlp_real_compare.py \
      --net small --device metal --mode allnode
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "auto_LiRPA"))

import numpy as np
import torch
import torch.nn as nn

from complete_verifier.api import ABCrownSolver, ConfigBuilder, VerificationSpec


# ---------------------------------------------------------------------------
# Networks with a scalar margin head.
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True)
        self.shortcut = (nn.Identity() if in_ch == out_ch and stride == 1
                         else nn.Conv2d(in_ch, out_ch, 1, stride, bias=False))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class CifarResNetMargin(nn.Module):
    """Conv stem -> residual layers -> avg pool -> FC head -> fixed margin
    layer. Final output is y_true - y_runnerup (one scalar)."""

    def __init__(self, channels=(8, 16, 32), blocks_per_layer=1, fc=64,
                 num_classes=10, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=True),
            nn.ReLU(),
        )
        layers = []
        in_ch = channels[0]
        for ci, ch in enumerate(channels):
            stride = 2 if ci > 0 else 1
            for b in range(blocks_per_layer):
                layers.append(ResBlock(in_ch, ch, stride=stride if b == 0 else 1))
                in_ch = ch
        self.body = nn.Sequential(*layers)
        # Global average pooling as a fixed Conv2d (kernel=stride=8):
        # BoundAvgPool.build_solver's shape check fails for kernel=stride,
        # and adaptive pooling has no LP solver support at all.
        pool_conv = nn.Conv2d(channels[-1], channels[-1], 8, 8, 0,
                              bias=False)
        with torch.no_grad():
            pool_conv.weight.copy_(
                torch.eye(channels[-1]).view(channels[-1], channels[-1], 1, 1) / 64.0)
        pool_conv.requires_grad_(False)
        self.pool = pool_conv
        self.fc1 = nn.Linear(channels[-1], fc)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc, num_classes)
        def _init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, generator=g)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)
        # Keep the margin O(1): kaiming-init logits are uncalibrated and
        # make the LP ill-conditioned. Scaling the weights avoids extra
        # Mul nodes in the computation graph (unsupported by the LP
        # solver builder).
        with torch.no_grad():
            self.fc2.weight.mul_(0.1)
            self.fc2.bias.mul_(0.1)

    def logits(self, x):
        out = self.stem(x)
        out = self.body(out)
        out = self.pool(out).flatten(1)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

    def set_margin_head(self, target, runner_up):
        """Replace the output with y_target - y_runner_up (fixed linear)."""
        with torch.no_grad():
            margin = nn.Linear(self.fc2.out_features, 1, bias=False)
            w = torch.zeros(1, self.fc2.out_features)
            w[0, target] = 1.0
            w[0, runner_up] = -1.0
            margin.weight.copy_(w)
        self.margin = margin
        return self

    def forward(self, x):
        return self.margin(self.logits(x))


def _fixed_margin_head(num_classes, target, runner_up):
    head = nn.Linear(num_classes, 1, bias=False)
    head.weight.data.zero_()
    head.weight.data[0, target] = 1.0
    head.weight.data[0, runner_up] = -1.0
    return head


def build_model(net_kind, seed=0):
    torch.manual_seed(seed)  # deterministic weights for all net kinds
    if net_kind == "smoke":
        net = nn.Sequential(
            nn.Conv2d(3, 4, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 10),
            _fixed_margin_head(10, 0, 1),
        )
        return net
    if net_kind == "tiny":
        # Minimal conv net for the all-node-split-LP leg: few enough
        # unstable ReLUs that domains reach the full-split LP phase.
        net = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(2, 2, 3, 2, 1, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 10),
            _fixed_margin_head(10, 0, 1),
        )
        return net
    if net_kind == "small":
        # Small conv net for the all-node-split-LP leg.
        net = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, 2, 1, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            _fixed_margin_head(10, 0, 1),
        )
        return net
    assert net_kind == "cifar-resnet"
    model = CifarResNetMargin(seed=seed)
    x0 = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(seed))
    logits = model.logits(x0)
    top2 = logits[0].topk(2).indices.tolist()
    return model.set_margin_head(top2[0], top2[1])


def build_spec(net_kind, seed, eps):
    torch.manual_seed(seed)
    shape = {"smoke": (1, 3, 8, 8), "tiny": (1, 1, 8, 8),
             "small": (1, 1, 16, 16),
             "cifar-resnet": (1, 3, 32, 32)}[net_kind]
    x0 = torch.randn(*shape) * 0.5
    lower = (x0 - eps).clamp(-3, 3)
    upper = (x0 + eps).clamp(-3, 3)
    clauses = [[(torch.tensor([[1.0]]), torch.tensor([0.0]))]]
    return VerificationSpec.build_from_input_bounds(lower, upper, clauses)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_config(mode, device, timeout, max_domains, alpha_iterations=20,
               fallback="none", mlxpdlp_tolerance=1e-4):
    b = ConfigBuilder()
    common = dict(
        general__device="cpu",
        general__seed=0,
        bab__timeout=timeout,
        bab__max_domains=max_domains,
        attack__pgd_order="skip",
        specification__norm=float("inf"),
        solver__mip__lp_backend="mlxpdlp",
        solver__mip__mlxpdlp_device=device,
        solver__mip__mlxpdlp_tolerance=mlxpdlp_tolerance,
        solver__mip__mlxpdlp_margin=1e-3,
        solver__mip__mlxpdlp_fallback=fallback,
        solver__mip__parallel_solvers=1,
        solver__mip__solver_threads=1,
        solver__mip__adv_warmup=False,
        solver__mip__skip_unsafe=True,
    )
    if mode == "allnode":
        # NOTE: the config key uses a HYPHEN ("beta-crown"), so it must be
        # passed positionally (it is not a valid Python keyword).
        b.set("solver__beta-crown__all_node_split_LP", True)
        b.set(general__complete_verifier="bab", **common)
    elif mode == "mip-refine":
        b.set(general__complete_verifier="mip",
              solver__mip__lp_solver=True,
              solver__mip__formulation="lp_integer", **common)
    elif mode == "bab-plain":
        b.set(general__complete_verifier="bab", **common)
    else:
        raise ValueError(mode)
    # "alpha-crown" contains a hyphen -> positional set().
    b.set("solver__alpha-crown__iteration", alpha_iterations)
    return b.to_dict()


def run_leg(mode, device, net_kind, seed, eps, timeout, max_domains,
            alpha_iterations=20, fallback="none", mlxpdlp_tolerance=1e-4):
    model = build_model(net_kind, seed=seed).eval()
    spec = build_spec(net_kind, seed, eps)
    config = make_config(mode, device, timeout, max_domains, alpha_iterations,
                         fallback, mlxpdlp_tolerance)
    solver = ABCrownSolver(spec, model, config=config)
    t0 = time.perf_counter()
    result = solver.verify()
    dt = time.perf_counter() - t0
    status = getattr(result, "status", None)
    glb = None
    try:
        glb = [float(v) for v in np.asarray(
            getattr(result, "global_lb", np.array([float("nan")]))).flatten()]
    except Exception:
        pass
    return {"mode": mode, "device": device, "net": net_kind,
            "time_sec": dt, "status": str(status), "global_lb": glb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="cifar-resnet",
                    choices=["smoke", "tiny", "small", "cifar-resnet"],
                    help="Network: smoke (pipeline validation), small "
                         "(all-node-split-LP leg), cifar-resnet (mip-refine "
                         "and baseline legs).")
    ap.add_argument("--device", default="metal", choices=["metal", "cpu"])
    ap.add_argument("--mode", default="allnode",
                    choices=["allnode", "mip-refine", "bab-plain"])
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-domains", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--mlxpdlp-tolerance", type=float, default=1e-4,
                    help="mlxPDLP optimality tolerance for the LP backend.")
    ap.add_argument("--fallback", default="none",
                    choices=["none", "cpu", "highs", "gurobi"],
                    help="mlxPDLP escalation fallback for the LP backend.")
    ap.add_argument("--alpha-iterations", type=int, default=20,
                    help="alpha-CROWN init iterations (100 default in the "
                         "verifier; 20 keeps the legs short)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="/tmp/abcrown_mlxpdlp_compare.json")
    args = ap.parse_args()

    print(f"=== leg: mode={args.mode} device={args.device} net={args.net} "
          f"eps={args.eps} timeout={args.timeout} max_domains={args.max_domains} ===",
          flush=True)
    rec = run_leg(args.mode, args.device, args.net, args.seed, args.eps,
                  args.timeout, args.max_domains, args.alpha_iterations,
                  args.fallback, args.mlxpdlp_tolerance)
    print("RESULT:", json.dumps(rec), flush=True)
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            records = json.load(f)
    else:
        records = []
    records.append(rec)
    with open(args.out_json, "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
