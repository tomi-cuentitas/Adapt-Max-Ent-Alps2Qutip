from copy import deepcopy
import numpy as np
import qutip as qutip
import scipy.linalg as linalg

### Parallelization functions employed using multithreading 

from itertools import product, combinations_with_replacement
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

### locally defined functions and operator classes

from .optimized_projections import opt_project_to_n_body_operator
from .operators.simplify import simplify_sum_operator

from alpsqutip.operators.arithmetic import (
    ScalarOperator,
    LocalOperator,
    OneBodyOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
    QutipOperator
)

### Workers functions at a ,,low"-level implementation, to be used in the 
### parallelization of future tasks.

def commutator_worker(pair):
    """Calcule [A, B] seulement si support(A) ∩ support(B) ≠ \null."""
    A, B = pair
    if frozenset(A.acts_over()).isdisjoint(B.acts_over()):
        return ScalarOperator(0, system=A.system)
    return simplify_sum_operator(A * B - B * A)

def general_worker(args):
    """
    Fonction polyvalente pour divers calculs parallèles.

    - mode='sp': (i, q, v, sp) → (i, sp(q, v))
    - mode='fine_sp': (i, j, q, v, sp) → (i, j, sp(q, v)) si supports communs
    - mode='projection': (term, nmax, sigma) → operator simplifié projeté
    - mode='hij': (i, b_i, b_last, H, sp, sigma, nmax) → (i, Re⟨b_i|[H, b_last]_proj⟩)

    """
    
    mode, data = args
    
    if mode == 'sp':
        i, q, v, sp = data
        return i, sp(q, v)

    elif mode == 'fine_sp':
        i, j, op1, op2, sp = data
        if frozenset(op1.acts_over()).isdisjoint(op2.acts_over()):
            return i, j, 0.0
        return i, j, sp(op1, op2)

    elif mode == 'projection':
        term, nmax, sigma_0 = data
        return (
            opt_project_to_n_body_operator(term, nmax=nmax, sigma=sigma_0)
            .simplify()
        )

    elif mode == 'hij':
        i, b_i, b_last, H, sp, sigma_0, nmax = data
        comm = commutator_worker((1j*H, b_last)).simplify()
        comm_proj = opt_project_to_n_body_operator(comm, nmax=nmax, sigma=sigma_0)
        return i, sp(b_i, comm_proj)

    else:
        raise ValueError(f"Mode inconnu: {mode}")

### 

from itertools import permutations
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collections import defaultdict

def parallelized_real_time_projection_of_hierarchical_basis(
    generator, 
    seed_op,
    sigma_ref,
    nmax,
    deep,
    num_workers=None,
    ell_prime=None,
    chunksize=None,
    tidy_thresh=1e-5,
):
    if seed_op is None or deep == 0:
        return []
    if ell_prime is None:
        ell_prime = deep

    basis = [seed_op]
    gen_terms = (-1j * generator).as_sum_of_products().terms
    system = generator.system

    for i in range(1, deep):
        current_op = basis[-1]
        basis_last_terms = (
            current_op.terms if hasattr(current_op, 'terms') 
            else current_op.as_sum_of_products().terms
        )

        gen_term_supports = {g: frozenset(g.acts_over()) for g in gen_terms}
        basis_term_supports = {b: frozenset(b.acts_over()) for b in basis_last_terms}

        term_pairs = [
            (g, b)
            for g in gen_terms
            for b in basis_last_terms
            if gen_term_supports[g].intersection(basis_term_supports[b])
        ]

        total_tasks = len(term_pairs)
        if chunksize is None:
            num_workers_eff = num_workers or os.cpu_count() or 4
            chunksize = max(1, total_tasks // (8 * num_workers_eff))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            commutator_terms = list(
                executor.map(commutator_worker, term_pairs, chunksize=chunksize)
            )

        grouped_terms = defaultdict(list)
        for term in commutator_terms:
            support = frozenset(term.acts_over())
            grouped_terms[support].append(term)

        merged_terms = [
            sum(group, ScalarOperator(0, system=system)) for group in grouped_terms.values()
        ]

        if i <= ell_prime:
            projection_args = [('projection', (term, nmax, sigma_ref)) for term in merged_terms]
        else:
            projection_args = [('projection', (term, 2, sigma_ref)) for term in merged_terms]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            projected_terms = list(
                executor.map(general_worker, projection_args, chunksize=chunksize)
            )

        local_op = sum(projected_terms, ScalarOperator(0, system=system))
        local_op = simplify_sum_operator(local_op)

        norm_val = local_op.to_qutip().norm()
        if norm_val < tidy_thresh:
            break

        basis.append(local_op)

    return basis

def compute_Hij_tensor_non_orth(
    basis, generator, sp, sigma_ref, nmax, Gram=None, num_workers=None, chunksize=None
):
    """
    Construct the Hij tensor with dynamic chunksize for parallelism.
    """
    n = len(basis)
    Hij_tensor = np.zeros((n, n), dtype=np.complex128)

    # Fill all columns except the last one
    if Gram is not None:
        for i in range(n):
            for j in range(n - 1):
                Hij_tensor[i, j] = Gram[i, j + 1]
    else:
        # Optional: parallelize this double loop if very large
        for i in range(n):
            for j in range(n - 1):
                Hij_tensor[i, j] = sp(basis[i], basis[j + 1])

    # Prepare args for the last column calculation
    total_tasks = n
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    if chunksize is None:
        # Create roughly 10 chunks per worker for good load balancing
        chunksize = max(1, total_tasks // (10 * num_workers))

    args = [
        ("hij", (i, basis[i], basis[-1], generator, sp, sigma_ref, nmax))
        for i in range(n)
    ]

    # Parallelize last column computation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(general_worker, args, chunksize=chunksize)

    for i, val in results:
        Hij_tensor[i, n - 1] = val

    return Hij_tensor

def orthogonalize_basis_parallel_process(
    basis, sp: callable, tol=1e-6, max_workers=None, return_orth_basis=False
):
    """
    Orthogonalizes a basis using Gram-Schmidt with parallel scalar product
    computation, and returns the transformation matrix and Gram matrix.

    Args:
        basis (List[Operator]): Initial operator basis {b_j}
        sp (callable): Scalar product function (must be picklable!)
        tol (float): Threshold to discard near-zero vectors
        max_workers (int or None): Max processes to use
        return_orth_basis (bool): If False, skips returning orthonormal basis

    Returns:
        orth_basis (List[Operator]) or None: If return_orth_basis is True
        R (np.ndarray): Transformation matrix, b_j = sum_i R[i,j] * q_i
        G (np.ndarray): Gram matrix, G = R† R
    """
    n = len(basis)
    system = getattr(basis[0], "system", None)

    def ensure_consistent(op):
        op.system = system
        if hasattr(op, "terms"):
            for t in op.terms:
                t.system = system
        return op

    orth_basis = []
    R = np.zeros((n, n), dtype=np.complex128)

    for j in range(n):
        v = deepcopy(basis[j])
        ensure_consistent(v)

        # Parallel scalar products <q_i, v>
        args = [(i, q, v, sp) for i, q in enumerate(orth_basis)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_scalar_product_task, args))

        for i, val in results:
            R[i, j] = val
            v = v - val * orth_basis[i]

        norm = np.real(sp(v, v)) ** 0.5
        if norm < tol:
            R[:, j] = 0.0
            continue

        R[len(orth_basis), j] = norm
        v = v / norm
        ensure_consistent(v)
        if return_orth_basis:
            orth_basis.append(v)

    R = R[:len(orth_basis), :]
    G = R.conj().T @ R

    return (orth_basis if return_orth_basis else None), R, G

def parallel_gram_matrix_process(basis, sp, num_workers=None, chunksize=16):
    """
    Computes the Hermitian Gram matrix G[i,j] = sp(b_i, b_j) in parallel using processes.

    Args:
        basis (List[Operator]): List of operator objects.
        sp (Callable): Pickleable scalar product function.
        num_workers (int or None): Number of parallel processes.

    Returns:
        G (np.ndarray): Hermitian Gram matrix.
    """
    n = len(basis)
    G = np.zeros((n, n), dtype=np.complex128)

    tasks = [(i, j, basis[i], basis[j], sp) for i in range(n) for j in range(i, n)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, j, val in executor.map(_fine_sp_worker, tasks, chunksize=16):
            G[i, j] = val
            if i != j:
                G[j, i] = np.conj(val)

    return G

def parallel_gram_matrix_fine(basis, sp, num_workers=None, chunksize=None):
    from collections import defaultdict
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    n = len(basis)
    flat_basis = [op.as_sum_of_products().terms for op in basis]

    tasks = [
       ("fine_sp", (i, j, a, b, sp))
        for i in range(n) for j in range(i, n)
        for a in flat_basis[i] for b in flat_basis[j]
    ]

    total_tasks = len(tasks)

    # Dynamically determine chunksize if not provided
    if chunksize is None:
        if num_workers is None:
            import multiprocessing
            num_workers = multiprocessing.cpu_count()
        chunksize = max(1, total_tasks // (10 * num_workers))  # heuristic: 10 chunks per worker

    partial_sums = defaultdict(complex)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, j, val in executor.map(general_worker, tasks, chunksize=chunksize):
            partial_sums[(i, j)] += val

    # Assemble final Gram matrix
    G = np.zeros((n, n), dtype=np.complex128)
    for (i, j), val in partial_sums.items():
        G[i, j] = val
        if i != j:
            G[j, i] = np.conj(val)

    return G

