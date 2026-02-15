import numpy as np
from pyomo.environ import *
from credit_model import run_credit_model
from vacancy_model import run_vacancy_model


def run_optimization():

    # ---------------- LOAD MODEL OUTPUTS ----------------
    credit_df = run_credit_model()
    listing_returns = run_vacancy_model()

    # ---------------- SAMPLING ----------------
    loan_sample = credit_df.sample(300, random_state=42).reset_index(drop=True)
    prop_sample = listing_returns.sample(100, random_state=42).reset_index(drop=True)

    loan_sample = loan_sample.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    prop_sample = prop_sample.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(loan_sample) == 0 or len(prop_sample) == 0:
        raise ValueError("Sampling produced empty dataset")

    # ---------------- PREPARE ARRAYS ----------------
    loan_capital = loan_sample["loan_amnt"].astype(float).values
    loan_return  = loan_sample["risk_adjusted_return"].astype(float).values

    prop_capital = (prop_sample["price"].astype(float) * 365).values
    prop_return  = prop_sample["risk_adjusted_return"].astype(float).values

    # ---------------- BUILD OPTIMIZATION MODEL ----------------
    model_opt = ConcreteModel()

    model_opt.L = RangeSet(0, len(loan_sample) - 1)
    model_opt.P = RangeSet(0, len(prop_sample) - 1)

    model_opt.xL = Var(model_opt.L, domain=Binary)
    model_opt.xP = Var(model_opt.P, domain=Binary)

    budget = 5_000_000

    model_opt.obj = Objective(
        expr=
        sum(model_opt.xL[i] * loan_return[i] for i in model_opt.L) +
        sum(model_opt.xP[j] * prop_return[j] for j in model_opt.P),
        sense=maximize
    )

    model_opt.budget = Constraint(
        expr=
        sum(model_opt.xL[i] * loan_capital[i] for i in model_opt.L) +
        sum(model_opt.xP[j] * prop_capital[j] for j in model_opt.P)
        <= budget
    )

    # ---------------- SOLVE ----------------
    solver = SolverFactory("glpk")
    result = solver.solve(model_opt)

    print("Solver Status:", result.solver.status)
    print("Termination Condition:", result.solver.termination_condition)

    if str(result.solver.termination_condition) != "optimal":
        raise ValueError("Optimization did not converge to optimal solution")

    # ---------------- EXTRACT RESULTS ----------------
    selected_loans = [i for i in model_opt.L if model_opt.xL[i].value == 1]
    selected_props = [j for j in model_opt.P if model_opt.xP[j].value == 1]

    total_capital = (
        sum(loan_capital[i] for i in selected_loans) +
        sum(prop_capital[j] for j in selected_props)
    )

    total_return = (
        sum(loan_return[i] for i in selected_loans) +
        sum(prop_return[j] for j in selected_props)
    )

    results = {
        "optimized_return": float(total_return),
        "capital_used": float(total_capital),
        "selected_loans": len(selected_loans),
        "selected_properties": len(selected_props)
    }

    return results


if __name__ == "__main__":
    results = run_optimization()
    print(results)
