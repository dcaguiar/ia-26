# ============================================================
# arquivo: main.py
# ============================================================

import matplotlib.pyplot as plt

import functions as f

# ============================================================
# IMPORTA METAHEURÍSTICAS
# ============================================================

from PSO import particle_swarm_optimization
from GA import genetic_algorithm
from DE import differential_evolution
from SA import simulated_annealing
from HC import hill_climbing


# ============================================================
# Plot convergência
# ============================================================

def plot_convergence(history, title):

    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.title(title)

    plt.xlabel("Iteração")
    plt.ylabel("Melhor valor")

    plt.yscale("log")

    plt.grid(True)

    plt.show()


# ============================================================
# Execução
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Escolha da função
    # ========================================================

    objective_function = f.f3

    bounds = (-32, 32)

    dim = 30

    # ========================================================
    # ESCOLHA METAHEURÍSTICA (descomente)
    # ========================================================

    # ============================================================
    # PSO
    # ============================================================

    # best_position, best_value, history = (
    #     particle_swarm_optimization(
    #         objective_function=objective_function,
    #         dim=dim,
    #         n_particles=100,
    #         max_iter=1000,
    #         bounds=bounds,
    #         w=0.7,
    #         c1=1.5,
    #         c2=1.5
    #     )
    # )


    # ============================================================
    # GA
    # ============================================================
    
    # best_position, best_value, history = (
    #     genetic_algorithm(
    #         objective_function=objective_function,
    #         dim=dim,
    #         population_size=50,
    #         generations=3000,
    #         bounds=bounds,
    #         crossover_rate=0.9,
    #         mutation_rate=0.1
    #     )
    # )


    # ============================================================
    # Differential Evolution
    # ============================================================

    
    # best_position, best_value, history = (
    #     differential_evolution(
    #         objective_function=objective_function,
    #         dim=dim,
    #         population_size=300,
    #         generations=350,
    #         bounds=bounds,
    #         F=0.5,
    #         CR=0.7
    #     )
    # )


    # ============================================================
    # Simulated Annealing
    # ============================================================

    best_position, best_value, history = (
        simulated_annealing(
            objective_function=objective_function,
            dim=dim,
            bounds=bounds,
            max_iter=1000,
            initial_temperature=10000,
            cooling_rate=0.995,
            step_size=1.0
        )
    )

    # ========================================================
    # Plot metaheurística
    # ========================================================

    plot_convergence(
        history,
        "Convergência Metaheurística"
    )

    print("\n===== METAHEURÍSTICA =====")

    print("Melhor posição:")
    print(best_position)

    print("\nMelhor valor:")
    print(best_value)

    # ========================================================
    # Hill Climbing
    # ========================================================

    hc_position, hc_value, hc_history = (
        hill_climbing(
            objective_function=objective_function,
            initial_solution=best_position,
            bounds=bounds,
            max_iter=5000,
            step_size=0.0001
        )
    )

    # ========================================================
    # Plot HC
    # ========================================================

    plot_convergence(
        hc_history,
        "Convergência Hill Climbing"
    )

    print("\n===== HILL CLIMBING =====")

    print("Melhor posição:")
    print(hc_position)

    print("\nMelhor valor:")
    print(hc_value)