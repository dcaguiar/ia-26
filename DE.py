import numpy as np
import matplotlib.pyplot as plt

import functions as f


# ============================================================
# Indivíduo
# ============================================================

class Individual:

    def __init__(
        self,
        dim,
        bounds,
        objective_function
    ):

        self.objective_function = objective_function

        self.vector = np.random.uniform(
            bounds[0],
            bounds[1],
            dim
        )

        self.fitness = self.objective_function(
            self.vector
        )

    def evaluate(self):

        self.fitness = self.objective_function(
            self.vector
        )


# ============================================================
# Differential Evolution
# ============================================================

def differential_evolution(
    objective_function,
    dim=30,
    population_size=100,
    generations=1000,
    bounds=(-100, 100),
    F=0.8,
    CR=0.9
):

    # ========================================================
    # Inicializa população
    # ========================================================

    population = [

        Individual(
            dim,
            bounds,
            objective_function
        )

        for _ in range(population_size)
    ]

    # ========================================================
    # Melhor solução global
    # ========================================================

    best_individual = min(
        population,
        key=lambda ind: ind.fitness
    )

    history = []

    # ========================================================
    # Loop principal
    # ========================================================

    for generation in range(generations):
        
        elite = min(
            population,
            key=lambda ind: ind.fitness
        )

        for i in range(population_size):

            target = population[i]

            # =================================================
            # Escolhe 3 indivíduos distintos
            # =================================================

            indices = list(range(population_size))

            indices.remove(i)

            a_idx, b_idx, c_idx = np.random.choice(
                indices,
                3,
                replace=False
            )

            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # =================================================
            # Mutação
            # v = a + F*(b - c)
            # =================================================

            mutant_vector = (
                a.vector
                + F * (b.vector - c.vector)
            )

            mutant_vector = np.clip(
                mutant_vector,
                bounds[0],
                bounds[1]
            )

            # =================================================
            # Crossover binomial
            # =================================================

            trial_vector = np.copy(target.vector)

            j_rand = np.random.randint(dim)

            for j in range(dim):

                if (
                    np.random.rand() < CR
                    or j == j_rand
                ):

                    trial_vector[j] = mutant_vector[j]

            # =================================================
            # Avalia vetor teste
            # =================================================

            trial_fitness = objective_function(
                trial_vector
            )

            # =================================================
            # Seleção
            # =================================================

            if trial_fitness < target.fitness:

                target.vector = trial_vector
                target.fitness = trial_fitness
            
        #ELITISMO    
        worst = max(
            population,
            key=lambda ind: ind.fitness
        )

        population.remove(worst)
        population.append(elite)

        # =====================================================
        # Melhor indivíduo da geração
        # =====================================================

        current_best = min(
            population,
            key=lambda ind: ind.fitness
        )

        if current_best.fitness < best_individual.fitness:

            best_individual = current_best

        history.append(best_individual.fitness)

        print(
            f"Geração {generation+1:4d}"
            f" | Melhor valor: "
            f"{best_individual.fitness:.10f}"
        )

    return (
        best_individual.vector,
        best_individual.fitness,
        history
    )


# ============================================================
# Plot convergência
# ============================================================

def plot_convergence(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.title("Convergência do Differential Evolution")

    plt.xlabel("Geração")
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
        differential_evolution(
            objective_function=objective_function,
            dim=30,
            population_size=300,
            generations=350,
            bounds=bounds,
            F=0.5,
            CR=0.7
        )
    )

    plot_convergence(history)

    print("\n===== RESULTADO FINAL =====")

    print("Melhor posição:")
    print(best_position)

    print("\nMelhor valor:")
    print(best_value)