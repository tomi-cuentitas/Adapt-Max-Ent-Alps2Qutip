"""
Functions to fetch specific scalar product functions.
"""

# from datetime import datetime
from typing import Callable

import numpy as np
from numpy import real

from alpsqutip.operators import Operator
from alpsqutip.operators.functions import anticommutator

#  ### Functions that build the scalar products ###


def fetch_kubo_scalar_product(sigma: Operator, threshold=0) -> Callable:
    """
    Build a KMB scalar product function
    associated to the state `sigma`
    """
    evals_evecs = sorted(zip(*sigma.eigenstates()), key=lambda x: -x[0])
    w = 1
    for i, val_vec in enumerate(evals_evecs):
        p = val_vec[0]
        w -= p
        if w < threshold or p <= 0:
            evals_evecs = evals_evecs[: i + 1]
            break

    def ksp(op1, op2):
        result = sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * (p1 if p1 == p2 else (p1 - p2) / np.log(p1 / p2))
            )
            for p1, v1 in evals_evecs
            for p2, v2 in evals_evecs
            if (p1 > 0 and p2 > 0)
        )

        #    stored[key] = result
        return result

    return ksp


def fetch_kubo_int_scalar_product(sigma: Operator) -> Callable:
    """
    Build a KMB scalar product function
    associated to the state `sigma`, from
    its integral form.
    """

    evals, evecs = sigma.eigenstates()

    def return_func(op1, op2):
        return 0.01 * sum(
            (
                np.conj((v2.dag() * op1 * v1).tr())
                * ((v2.dag() * op2 * v1).tr())
                * ((p1) ** (1.0 - tau))
                * ((p1) ** (tau))
            )
            for p1, v1 in zip(evals, evecs)
            for p2, v2 in zip(evals, evecs)
            for tau in np.linspace(0.0, 1.0, 100)
            if (p1 > 0.0 and p2 > 0.0)
        )

    return return_func

from functools import partial
from numpy import real

def _covar_sp(op1: Operator, op2: Operator, sigma: Operator):
    op1_herm = op1.isherm
    op2_herm = op2.isherm
    if op1_herm:
        if op2_herm:
            return real(sigma.expect(op1 * op2))
        op1_dag = op1
    else:
        op1_dag = op1.dag()
    if op1_dag is op2:
        return sigma.expect((op1_dag * op2).simplify())
    else:
        return 0.5 * sigma.expect(anticommutator(op1_dag, op2))

def fetch_covar_scalar_product(sigma: Operator):
    """
    Returns a scalar product function based on the covariance of a density
    operator. Pickle-safe version.
    """
    return partial(_covar_sp, sigma=sigma)



def fetch_HS_scalar_product() -> Callable:
    """
    Build a HS scalar product function
    """
    return lambda op1, op2: (op1.dag() * op2).tr()