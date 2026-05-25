# ============================================================
# arquivo: hill_climbing.py
# ============================================================

import numpy as np


# ============================================================
# Hill Climbing
# ============================================================

def hill_climbing(
    objective_function,
    initial_solution,
    bounds,
    max_iter=5000,
    step_size=0.001
):

    # ========================================================
    # Solução inicial
    # ========================================================

    current_solution = initial_solution.copy()

    current_value = objective_function(
        current_solution
    )

    best_solution = current_solution.copy()

    best_value = current_value

    history = [best_value]

    # ========================================================
    # Loop principal
    # ========================================================

    for iteration in range(max_iter):

        # ====================================================
        # Gera vizinho
        # ====================================================

        neighbor = (
            current_solution
            + np.random.normal(
                0,
                step_size,
                len(current_solution)
            )
        )

        neighbor = np.clip(
            neighbor,
            bounds[0],
            bounds[1]
        )

        neighbor_value = objective_function(
            neighbor
        )

        # ====================================================
        # Aceita apenas melhoria
        # ====================================================

        if neighbor_value < current_value:

            current_solution = neighbor
            current_value = neighbor_value

            if current_value < best_value:

                best_solution = (
                    current_solution.copy()
                )

                best_value = current_value

        history.append(best_value)

    return (
        best_solution,
        best_value,
        history
    )