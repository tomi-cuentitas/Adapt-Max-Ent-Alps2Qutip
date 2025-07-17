# adaptive_maxent/system_setup.py

import numpy as np
from functools import reduce

from alpsqutip import build_system
from alpsqutip.operators.arithmetic import Operator
from alpsqutip.operators.states.gibbs import GibbsProductDensityOperator


def build_spin_chain(params):
    """Builds the spin system and returns all relevant initial state data."""
    size = params["size"]
    system = build_system(geometry_name="open chain lattice", model_name="spin", L=size, J=1)

    sites = list(system.sites)
    sx_ops = [system.site_operator("Sx", f"1[{i}]") for i in range(size)]
    sy_ops = [system.site_operator("Sy", f"1[{i}]") for i in range(size)]
    sz_ops = [system.site_operator("Sz", f"1[{i}]") for i in range(size)]

    idop = reduce(Operator.__mul__, [system.site_operator("identity", f"1[{i}]") for i in range(size)])

    # Define XYZ Hamiltonian
    H = sum(
        params["Jx"] * sx_ops[i] * sx_ops[i + 1] +
        params["Jy"] * sy_ops[i] * sy_ops[i + 1] +
        params["Jz"] * sz_ops[i] * sz_ops[i + 1]
        for i in range(size - 1)
    ).simplify()

    # Define basis and initial state (Gibbs product)
    HBB0 = [idop, sx_ops[0], sy_ops[0], sz_ops[0]]
    phi0 = np.array([0.0, 0.25, 0.25, 10.0])
    K0 = phi0 @ HBB0
    sigma0 = GibbsProductDensityOperator(K0)
    phi0[0] = np.log(sigma0.tr())
    K0 = phi0 @ HBB0
    sigma0 = GibbsProductDensityOperator(K0)

    obs_Sz_total = sum(sz_ops)

    return {
        "idop": idop,
        "system": system,
        "sx_ops": sx_ops,
        "sy_ops": sy_ops,
        "sz_ops": sz_ops,
        "H": H,
        "idop": idop,
        "K0": K0,
        "sigma0": sigma0,
        "obs": obs_Sz_total
    }
