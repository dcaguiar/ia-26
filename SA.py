import numpy as np
import matplotlib.pyplot as plt

import functions as f


# ============================================================
# Simulated Annealing
# ============================================================

def simulated_annealing(
    objective_function,
    dim=30,
    bounds=(-100, 100),
    max_iter=10000,
    initial_temperature=1000,
    cooling_rate=0.995,
    step_size=1.0
):

    # ========================================================
    # Solução inicial
    # ========================================================

    current_solution = np.random.uniform(
        bounds[0],
        bounds[1],
        dim
    )

    current_value = objective_function(
        current_solution
    )

    # ========================================================
    # Melhor solução global
    # ========================================================

    best_solution = current_solution.copy()

    best_value = current_value

    # ========================================================
    # Temperatura inicial
    # ========================================================

    temperature = initial_temperature

    history = []

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
                dim
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
        # Diferença de energia
        # ====================================================

        delta = neighbor_value - current_value

        # ====================================================
        # Critério de aceitação
        # ====================================================

        if delta < 0:

            # Melhor solução → aceita
            current_solution = neighbor
            current_value = neighbor_value

        else:

            # Aceita solução pior
            probability = np.exp(
                -delta / temperature
            )

            if np.random.rand() < probability:

                current_solution = neighbor
                current_value = neighbor_value

        # ====================================================
        # Atualiza melhor global
        # ====================================================

        if current_value < best_value:

            best_solution = (
                current_solution.copy()
            )

            best_value = current_value

        # ====================================================
        # Resfriamento
        # ====================================================

        temperature *= cooling_rate

        history.append(best_value)

        print(
            f"Iteração {iteration+1:5d}"
            f" | Melhor valor: "
            f"{best_value:.10f}"
            f" | Temperatura: "
            f"{temperature:.5f}"
        )

    return (
        best_solution,
        best_value,
        history
    )


# ============================================================
# Plot convergência
# ============================================================

def plot_convergence(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.title("Convergência do Simulated Annealing")

    plt.xlabel("Iteração")
    plt.ylabel("Melhor valor")

    plt.yscale("log")

    plt.grid(True)

    plt.show()


# ============================================================
# Execução
# ============================================================

if __name__ == "__main__":

    objective_function = f.f1

    bounds = (-100, 100)

    best_position, best_value, history = (
        simulated_annealing(
            objective_function=objective_function,
            dim=30,
            bounds=bounds,
            max_iter=10000,
            initial_temperature=1000,
            cooling_rate=0.995,
            step_size=1.0
        )
    )

    plot_convergence(history)

    print("\n===== RESULTADO FINAL =====")

    print("Melhor posição:")
    print(best_position)

    print("\nMelhor valor:")
    print(best_value)