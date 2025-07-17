"""
Basic unit test.
"""

from alpsqutip.operators import (
    LocalOperator,
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from alpsqutip.operators.quadratic import QuadraticFormOperator
from alpsqutip.operators.simplify import group_terms_by_blocks, simplify_sum_operator
from alpsqutip.operators.states import GibbsDensityOperator, GibbsProductDensityOperator

from .helper import OPERATORS, check_equality, check_operator_equality, full_test_cases


def compute_size(operator: Operator):
    """
    compute the initial number of
    qutip operators needed to store
    operator
    """
    if isinstance(operator, ScalarOperator):
        return 0
    if isinstance(operator, LocalOperator):
        return 1
    if isinstance(operator, ProductOperator):
        return len(operator.sites_op)
    if isinstance(operator, SumOperator):
        return sum(compute_size(term) for term in operator.terms)
    if isinstance(operator, QutipOperator):
        return 1
    if isinstance(operator, QuadraticFormOperator):
        return sum(compute_size(term) for term in operator.basis)
    if isinstance(operator, GibbsProductDensityOperator):
        return len(operator.k_by_site)
    if isinstance(operator, GibbsDensityOperator):
        return compute_size(operator.k)
    raise ValueError(f"Unknown kind of operator {type(operator)}")


def test_simplify():
    """test simplify operators"""

    passed = True
    for key, operator in full_test_cases.items():
        print("* check", key)
        simplify1 = operator.simplify()
        if not (check_operator_equality(operator, simplify1)):
            print("    1. simplify changed the value of the operator")
            passed = False
            continue

        try:
            cases_dict = {"square": operator * operator, "sum": operator + operator}
        except ValueError:
            continue

        for arith_op, op_test in cases_dict.items():
            initial_size = compute_size(op_test)
            print("    checking with ", arith_op, " which produced", type(op_test))
            type_operand = type(op_test)
            simplify1 = op_test.simplify()
            simplify2 = simplify1.simplify()
            if type(simplify1) is not type(simplify2):
                print(" types do not match")
                passed = False
                continue

            if isinstance(simplify1, SumOperator):
                print("        checking the consistency of sum operators")
                if len(simplify1.terms) != len(simplify2.terms):
                    print(
                        "  number of terms do not match with the",
                        "first simplification",
                        len(simplify1.terms),
                        "!=",
                        len(simplify2.terms),
                    )
                    passed = False
                    continue
                if not all(
                    check_operator_equality(t1, t2)
                    for t1, t2 in zip(simplify1.terms, simplify2.terms)
                ):
                    print("different terms obtained")
                    passed = False
                    continue

                for t1, t2 in zip(simplify1.terms, simplify2.terms):
                    if t1 is not t2:
                        passed = False
                        print(f"{t1} is not {t2}")
                        continue

            print("        checking fixed point")
            if simplify1 is not simplify2:
                passed = False
                print("simplify should reach a fix point.", f"{simplify1}->{simplify2}")
                continue

            print("        checking properties")
            # assert op_test.isherm == simplify1.isherm,
            # "hermiticity should be preserved"
            if not (simplify1.isdiagonal or not op_test.isdiagonal):
                print("      diagonality should be preserved")
                passed = False
                continue

            print("        checking that indeed the expression was simplified")
            if isinstance(op_test, SumOperator):
                if isinstance(simplify1, SumOperator):
                    final_size = compute_size(simplify1)
                    print("                - final size of the operator:", final_size)
                    if not (initial_size >= final_size):
                        print(
                            "we should get less terms, not more ",
                            f"({initial_size} < {final_size}).",
                        )
                        passed = False
                        continue
                else:
                    if not isinstance(
                        simplify1,
                        (
                            type_operand,
                            ScalarOperator,
                            LocalOperator,
                            ProductOperator,
                            QutipOperator,
                        ),
                    ):
                        print("   resunting type is not valid ", f"({type(simplify1)})")
                        passed = False
                        continue
    assert not passed, "there were errors in simplificacion."


def test_simplify_sum_operator():
    def do_test(name, operator):
        if isinstance(operator, list):
            for op_case in operator:
                do_test(name, op_case)
            return
        operator_simpl = simplify_sum_operator(operator)
        assert check_equality(operator.to_qutip(), operator_simpl.to_qutip())
        assert operator.to_qutip().isherm == operator_simpl.isherm

    for name, operator_case in OPERATORS.items():
        print("name", name, type(operator_case))
        do_test(name, operator_case)


def test_sum_as_blocks():
    print("Sum as blocks")
    for key, operator in full_test_cases.items():
        print(f"   checking {key}")
        operator_sab = group_terms_by_blocks(
            operator, fn=lambda x: x.to_qutip_operator()
        )
        if operator_sab is operator:
            continue
        assert check_operator_equality(operator_sab, operator)
        if not isinstance(operator_sab, SumOperator):
            continue
        acts_over_lst = [frozenset(term.acts_over()) for term in operator_sab.terms]
        acts_over_set = set(acts_over_lst)
        assert all(
            block is None or len(block) != 1 for block in acts_over_set
        ), f"One body terms should be together. Found {acts_over_set}"
        assert len(acts_over_set) == len(acts_over_lst), "Repeated blocks found"
