import numpy as np
import qutip as qutip
import scipy.linalg as linalg
import time



from alpsqutip import (build_system, list_models_in_alps_xml,
                       list_geometries_in_alps_xml, graph_from_alps_xml,
                       model_from_alps_xml,
                       restricted_maxent_toolkit as me)

from alpsqutip.operators.states.utils import safe_exp_and_normalize ## function used to safely and robustly map K-states to states

from alpsqutip.operators.states.meanfield.projections import project_operator_to_m_body
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

from alpsqutip.scalarprod import (gram_matrix, fetch_covar_scalar_product, 
                                  orthogonalize_basis, orthogonalize_basis_gs)

import logging

def real_time_projection_of_hierarchical_basis(generator, 
                                               seed_op,
                                               sigma_ref,
                                               nmax, 
                                               deep):
    basis = []
    if seed_op is not None and deep > 0:
        basis += [seed_op]
        
        for i in range(1, deep):
            local_op = -1j*me.commutator(generator.to_qutip_operator(), basis[-1].to_qutip_operator())
            if i <= int(deep/3):
                basis.append(project_to_n_body_operator(operator = local_op.tidyup(1e-5).as_sum_of_products(),
                                                      nmax = nmax, 
                                                      sigma = sigma_ref).tidyup(1e-5))
            else:
                basis.append(project_to_n_body_operator(operator = local_op.tidyup(1e-5).as_sum_of_products(),
                                                      nmax = nmax, 
                                                      sigma = sigma_ref).tidyup(1e-5))
            local_op = None
            
    return basis

def run_adaptive_simulation(params, tgt_obs, ):
    
    """
    params : a dictionary of the system properties (Hamiltonian) and of the simulation parameters
    """
    
    H = 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
