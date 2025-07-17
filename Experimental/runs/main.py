# Experimental/runs/main.py

import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import fsolve

from .system_setup import build_spin_chain
from .simulation import run_simulation

def estimate_vLR(params):
    Jx, Jy, Jz = params["Jx"], params["Jy"], params["Jz"]
    coef = [1, 0, -(Jx * Jy + Jx * Jz + Jy * Jz), -2 * Jx * Jy * Jz]
    roots = np.roots(coef)
    Ffactor = np.real(max(roots))
    chi_y = fsolve(lambda x, y: x * np.arcsinh(x) - np.sqrt(x**2 + 1) - y, 1e-1, args=(0))[0]
    return 4 * Ffactor * chi_y

def parse_args():
    ap = argparse.ArgumentParser(description="Adaptive Max-Ent simulation")
    ap.add_argument("--size", type=int, default=7, help="System size")
    ap.add_argument("--Jx", type=float, default=1.25, help="Jx coupling")
    ap.add_argument("--Jy", type=float, default=1.0, help="Jy coupling")
    ap.add_argument("--Jz", type=float, default=0.0, help="Jz coupling")
    ap.add_argument("--t_max", type=float, default=70.0, help="Maximum simulation time")
    ap.add_argument("--steps", type=int, default=100, help="Number of time steps")
    ap.add_argument("--save_every", type=int, default=10, help="Save checkpoint every n steps")
    ap.add_argument("--out", type=str, default="checkpoints", help="Output directory")
    ap.add_argument("--depth", type=int, default=6, help="Max commutator depth (ell)")
    ap.add_argument("--m0", type=int, default=2, help="Max n-body range in HBB basis")
    ap.add_argument("--eps", type=float, default=1e-4, help="Error threshold for adaptive updates")
    ap.add_argument("--num_workers", type=int, default=2, help="Parallel worker pool size")
    #ap.add_argument('--count_basis', action='store_true',
    #               help='Track |orthonormal product basis| of K(t) and write it '
    #                    'to the CSV (extra column orth_len).')
    return ap.parse_args()

def main():
    args = parse_args()

    # Physics parameters for system construction
    physics_params = dict(
        size=args.size,
        Jx=args.Jx,
        Jy=args.Jy,
        Jz=args.Jz,
    )

    # Simulation hyperparameters
    sim_hyper = dict(
        chosen_depth=args.depth,
        m0=args.m0,
        eps=args.eps,
        ell_prime=None,
    )

    # Estimate Lieb-Robinson velocity scaling factor
    vLR = estimate_vLR(physics_params)
    timespan = np.linspace(0.0, args.t_max, args.steps)

    print(f"Starting simulation with parameters: size={args.size}, Jx={args.Jx}, Jy={args.Jy}, Jz={args.Jz}")
    print(f"Simulation timespan: 0 to {args.t_max} (scaled by vLR={vLR:.4f}) in {args.steps} steps")

    # Build system objects
    system_objects = build_spin_chain(physics_params)

    # Combine all parameters for run_simulation
    full_params = {**physics_params, **sim_hyper}

    # Run the simulation
    run_simulation(
        sim_id=1,
        params=full_params,
        system_objects=system_objects,
        timespan=timespan,
        save_every=args.save_every,
        out_dir=Path(args.out),
        num_workers=args.num_workers
        #count_basis=args.count_basis
    )

if __name__ == "__main__":
    main()
