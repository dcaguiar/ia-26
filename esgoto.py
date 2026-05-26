import numpy as np

from GA import genetic_algorithm


flows = np.array([
    2.5, 5.0, 7.5, 10.0, 12.5,
    15.0, 17.5, 20.0, 22.5, 25.0
]) / 1000  # m³/s

lengths = np.array([
    20, 54, 98, 120, 34,
    12, 88, 122, 33, 40
])

diameters = np.array([
    0.150,
    0.200,
    0.250,
    0.300,
    0.400
])

costs = np.array([
    65,
    98,
    150,
    210,
    340
])

# Constantes
n = 0.013
S = 0.005

def manning_flow(D):

    A = (
        np.pi * D**2
    ) / 4

    Rh = D / 4

    Qmax = (
        (1 / n)
        * A
        * (Rh ** (2/3))
        * np.sqrt(S)
    )

    return Qmax


def objective_function(solution):
    
    # Converte para inteiro
    solution = np.round(solution).astype(int)

    # Garante limites
    solution = np.clip(
        solution,
        0,
        len(diameters) - 1
    )

    # Seleciona tubos
    selected_diameters = (
        diameters[solution]
    )

    selected_costs = (
        costs[solution]
    )

    # Custo total
    total_cost = np.sum(
        selected_costs * lengths
    )

    # Penalização
    penalty = 0

    for i in range(len(solution)):
        
        D = selected_diameters[i]
        Q = flows[i]
        Qmax = manning_flow(D)


        # Restrição:
        # Q <= 75% da capacidade
        if Q > 0.75 * Qmax:
            penalty += 1e9

    return total_cost + penalty



if __name__ == "__main__":

    best_position, best_value, history = (
        genetic_algorithm(
            objective_function=objective_function,
            dim=10,
            population_size=50,
            generations=100,
            bounds=(0, 4),
            crossover_rate=0.9,
            mutation_rate=0.1
        )
    )

    # Converte solução final
    best_position = np.round(
        best_position
    ).astype(int)

    best_position = np.clip(
        best_position,
        0,
        4
    )


    # Resultado
    print("\n===== MELHOR SOLUÇÃO =====\n")

    total_cost = 0

    for i in range(10):
        
        d = diameters[
            best_position[i]
        ]
        
        c = costs[
            best_position[i]
        ]

        trecho_cost = (
            c * lengths[i]
        )

        total_cost += trecho_cost

        print(
            f"Trecho {i+1:2d}"
            f" | D = {int(d*1000):3d} mm"
            f" | Custo = R$ {trecho_cost:.2f}"
        )

    print("\n==========================")

    print(
        f"\nCusto total: "
        f"R$ {total_cost:.2f}"
    )

    print(
        f"\nFitness final: "
        f"{best_value}"
    )