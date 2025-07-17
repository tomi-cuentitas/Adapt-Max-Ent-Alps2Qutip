import numpy as np
import scipy.linalg as linalg
from pathlib import Path
from typing import List

from qutip import mesolve
from alpsqutip import restricted_maxent_toolkit as me
from alpsqutip.operators.states.gibbs import GibbsDensityOperator, GibbsProductDensityOperator
from alpsqutip.scalarprod import fetch_covar_scalar_product
import alpsqutip.parallelized_functions_and_workers as plME

from .io_utils import save_checkpoint
from scipy.optimize import fsolve

# -----------------------------------------------------------------------------
# helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def Heisenberg_vLR(params):
    Jx = params['Jx']
    Jy = params['Jy']
    Jz = params['Jz']
    
    poly_coeffs = [1, 0, -(Jx*Jy + Jx*Jz + Jy*Jz), -2*Jx*Jy*Jz]
    roots = np.roots(poly_coeffs)
    Ffactor = np.max(np.real(roots[np.isreal(roots)]))

    chi_y = fsolve(lambda x: x*np.arcsinh(x) - np.sqrt(x**2 + 1), 0.1)[0]
    
    return 4 * Ffactor * chi_y

def flatten_to_product_terms(op):
    """Recursively collect all ProductOperator terms inside *op*."""
    if hasattr(op, "as_sum_of_products"):
        sop = op.as_sum_of_products()
        if hasattr(sop, "terms"):
            return [t for sub in sop.terms for t in flatten_to_product_terms(sub)]
    return [op]


def orthogonalize_operators(ops, sp, tol: float = 1e-12):
    """Return an orthonormal list of *ops* with respect to scalar product *sp*."""
    n = len(ops)
    G = np.empty((n, n), dtype=complex)
    for i in range(n):
        for j in range(i, n):
            G[i, j] = sp(ops[i], ops[j])
            G[j, i] = np.conjugate(G[i, j])
    eigvals, U = np.linalg.eigh(G)
    keep = eigvals > tol
    if not keep.any():
        raise RuntimeError("operator set is linearly dependent")
    C = U[:, keep] @ np.diag(1.0 / np.sqrt(eigvals[keep]))
    return [sum(C[i, k] * ops[i] for i in range(n)) for k in range(C.shape[1])]

from multiprocessing import Pool

def project_flat_list(flat_list, sigma_act, num_workers=4):
    """Project each operator in flat_list using parallel workers."""
    args = [("projection", (op, 2, sigma_act)) for op in flat_list]
    with Pool(processes=num_workers) as pool:
        projected = pool.map(plME.general_worker, args)
    return projected

# -----------------------------------------------------------------------------
# main driver -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def run_simulation(
    sim_id: int,
    params: dict,
    system_objects: dict,
    timespan: np.ndarray,
    save_every: int,
    out_dir,
    *,
    num_workers: int = 2
):
    """Adaptive Heisenberg‐picture simulation with optional basis diagnostics.

    Columns saved per checkpoint:
      * t
      * ev          – approximate expectation value
      * ev_exact    – exact (qutip) expectation value
      * orth_len    – |orth basis| (every `basis_step` if `count_basis`)
      * ints_err    – instantaneous ∫ ||[H, O_last]|| bound (every step)
    """

    tag = (
        f"Jx{params['Jx']}_Jy{params['Jy']}_Jz{params['Jz']}"
        f"_ell{params['chosen_depth']}_m{params['m0']}_eps{params['eps']}"
    )
    
    sim_out_dir = Path(out_dir) / tag
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Simulation outputs will be saved to: {sim_out_dir}")

    idop, H, K0, sigma0, tgt_obs = (
        system_objects["idop"],
        system_objects["H"],
        system_objects["K0"],
        system_objects["sigma0"],
        system_objects["obs"],
    )

    vLR = Heisenberg_vLR(params)
    res_exact = mesolve(H.to_qutip(), sigma0.to_qutip(), tlist=timespan, e_ops=[tgt_obs.to_qutip()])
    ev_exact_arr = np.asarray(res_exact.expect[0], dtype=complex)

    sim = {
        "evs": [(sigma0 * tgt_obs).tr()],
        "evs_exact": [ev_exact_arr[0]]
    }

    sigma_act = sigma0
    sp_local = fetch_covar_scalar_product(sigma0)
    chosen_depth, m0 = params["chosen_depth"], params["m0"]

    HBB = plME.parallelized_real_time_projection_of_hierarchical_basis(
        generator=H,
        seed_op=K0 + 1e-10 * H,
        sigma_ref=sigma0,
        nmax=m0,
        deep=chosen_depth,
        num_workers=num_workers
    )
    
    HBB = [op.tidyup(1e-7) for op in HBB]
   
    Gram = plME.parallel_gram_matrix_fine(HBB, sp_local, num_workers=num_workers)
    R = np.linalg.cholesky(Gram).conj().T
    b_orth = np.linalg.inv(R.T) @ HBB

    Hij = np.linalg.inv(R).T @ plME.compute_Hij_tensor_non_orth(
        basis=HBB, generator=H, sp=sp_local, sigma_ref=sigma0,
        nmax=m0, Gram=Gram, num_workers=num_workers) @ np.linalg.inv(R)
    phi0 = np.array([sp_local(K0, op) for op in b_orth])
    
    recorded_times: List[float] = [timespan[0]]
    for idx, t in enumerate(timespan[1:], start=1):
        dt = t - recorded_times[-1]
        phi0 = np.real(linalg.expm(dt * Hij) @ phi0)       # <‑‑ advance state *and* store back
        K_local = (phi0 @ b_orth).simplify().tidyup(1e-7)
        ev_val = (GibbsDensityOperator(K_local) * tgt_obs).tr()
        
        sim["evs"].append(ev_val)
        sim["evs_exact"].append(ev_exact_arr[idx] if idx < len(ev_exact_arr) else np.nan)
        recorded_times.append(t)

        if idx % max(1, len(timespan)//10) == 0 or idx == len(timespan)-1:
            print(f"[PROGRESS] t = {t:.3f} | ev ≈ {ev_val.real:.6f} | ev_exact = {ev_exact_arr[idx].real:.6f}")

        err_ratio = me.m_th_partial_sum(phi=phi0, m=2) / me.m_th_partial_sum(phi=phi0, m=0)
        if abs(err_ratio) >= params["eps"]:
            print(f"[INFO] Error threshold exceeded (err={err_ratio:.3e}), rebuilding basis at t={t:.3f}")
            print(f"        ev ≈ {ev_val.real:.6f} | ev_exact = {ev_exact_arr[idx].real:.6f}")

            HBB = plME.parallelized_real_time_projection_of_hierarchical_basis(
                generator=H, seed_op=K_local, sigma_ref=sigma0,
                nmax=m0, deep=chosen_depth, num_workers=num_workers)
            
            HBB = [op.tidyup(1e-7) for op in HBB]

            Gram = plME.parallel_gram_matrix_fine(HBB, sp_local, num_workers=num_workers)
            R = np.linalg.cholesky(Gram).conj().T
            b_orth = np.linalg.inv(R.T) @ HBB

            Hij = np.linalg.inv(R).T @ plME.compute_Hij_tensor_non_orth(
                basis=HBB, generator=H, sp=sp_local, sigma_ref=sigma0,
                nmax=m0, Gram=Gram, num_workers=num_workers) @ np.linalg.inv(R)

            phi0 = np.array([sp_local(K_local, op) for op in b_orth])
            
            
        L = len(recorded_times)
        for key in ["evs", "evs_exact"]:
            if key in sim:
                while len(sim[key]) < L:
                    sim[key].append(np.nan)

        if idx % save_every == 0 or idx == len(timespan) - 1:
            save_checkpoint(
                sim_id,
                recorded_times,
                sim["evs"],
                evs_exact=sim["evs_exact"],
                out_dir=sim_out_dir,
                step=idx
            )