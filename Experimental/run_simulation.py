# run_simulation.py

import numpy as np
import matplotlib.pyplot as plt
import qutip as qutip
import scipy.linalg as linalg
import time

import sys
path='/home/tomas/Ulmer-Berechnung/alps2qutipplus-april/alps2qutipplus-main/'

sys.path.insert(1, path) 

from alpsqutip import (build_system, list_models_in_alps_xml,
                       list_geometries_in_alps_xml, graph_from_alps_xml,
                       model_from_alps_xml,
                       restricted_maxent_toolkit as me)

from alpsqutip.operators.states.utils import safe_exp_and_normalize 
from alpsqutip.operators.states.meanfield.projections import (one_body_from_qutip_operator, 
                                                  project_operator_to_m_body, 
                                                             project_qutip_operator_to_m_body,
                                                             project_to_n_body_operator,
                                                             project_qutip_operator_as_n_body_operator)

from alpsqutip.operators import (
    ScalarOperator,
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
    QutipOperator
)

from alpsqutip.scalarprod import gram_matrix, fetch_covar_scalar_product, orthogonalize_basis, orthogonalize_basis_gs
from alpsqutip.operators.states.gibbs import GibbsDensityOperator, GibbsProductDensityOperator
from Experimental.projs_and_actualizations import heisenberg_actualize_and_project_basis, simulate_projected_evolution
import logging
import json
import os
import time

from scipy.optimize import root, fsolve
from itertools import combinations

def setup_logging(log_dir="logs", tag="longrange_heisenberg"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"{tag}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

def save_results(results, tag="results", base_dir="data"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(base_dir, f"{tag}_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {filename}")

def setup_system_and_initial_state():
    params = {}
    params['size'] = 5
    params['Jx'] = 1.0
    params['Jy'] = 0.75 * params['Jx']
    params['Jz'] = 1.05 * params['Jx']

    # Lieb-Robinson velocity
    Ffactor = np.real(max(np.roots(np.poly1d([
        1, 0,
        -(params['Jx']*params['Jy'] + params['Jx']*params['Jz'] + params['Jy']*params['Jz']),
        -2*params['Jx']*params['Jy']*params['Jz']
    ]))))
    chi_y = fsolve(lambda x, y: x*np.arcsinh(x) - np.sqrt(x**2 + 1) - y, 1e-1, args=(0))[0]
    vLR = 4 * Ffactor * chi_y

    # Build system
    system = build_system("open chain lattice", "spin", L=params['size'], J=1)
    sites = system.sites
    sx_ops = [system.site_operator("Sx", f"1[{a}]") for a in range(len(sites))]
    sy_ops = [system.site_operator("Sy", f"1[{a}]") for a in range(len(sites))]
    sz_ops = [system.site_operator("Sz", f"1[{a}]") for a in range(len(sites))]

    # Identity operator
    idop = system.site_operator(f'identity@1[0]')
    for i in range(1, params['size']):
        idop *= system.site_operator(f'identity@1[{i}]')

    # Construct Hamiltonian
    H_nn, H_lr = 0, 0
    for i, j in combinations(range(params['size']), 2):
        r = abs(i - j)
        Jx_ij = params['Jx'] / r**3
        Jy_ij = params['Jy'] / r**3
        Jz_ij = params['Jz'] / r**3
        term = Jx_ij * sx_ops[i]*sx_ops[j] + Jy_ij * sy_ops[i]*sy_ops[j] + Jz_ij * sz_ops[i]*sz_ops[j]
        if r == 1:
            H_nn += term
        else:
            H_lr += term

    HBB0 = [idop, sz_ops[2], H_nn]
    phi0 = np.array([0.0, -0.75, 0.3])
    K0 = me.k_state_from_phi_basis(phi0, HBB0)
    sigma0 = GibbsDensityOperator(K0)
    phi0[0] = np.log(sigma0.tr())
    K0 = me.k_state_from_phi_basis(phi0, HBB0).simplify()
    sigma0 = GibbsDensityOperator(K0)

    obs = sum(sz_ops)
    timespan = np.linspace(0.0, 200.1/vLR, 30)
    
    H = H_nn+H_lr
    parms = {
        "HBB0": HBB0,
        "Jx": params['Jx'],
        "Jy": params['Jy'],
        "Jz": params['Jz'],
        "size": params['size'],
        "chosen_depth": 5,
        "m0":3,
        "eps": 1e-3
    }

    return sigma0, H, obs, timespan, parms

if __name__ == "__main__":
    log_file = setup_logging(tag="longrange_heisenberg")
    sigma0, H, obs, timespan, parms = setup_system_and_initial_state()

    logging.info("Starting projected evolution simulation...")
    results = simulate_projected_evolution(
        generator=H,
        sigma0=sigma0,
        tgt_obs=obs,
        timespan=timespan,
        parms=parms,
        log_interval=10,
    )
   
    save_results(results, tag="longrange_heisenberg")
    logging.info("Simulation and saving completed.")