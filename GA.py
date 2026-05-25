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

        self.chromosome = np.random.uniform(
            bounds[0],
            bounds[1],
            dim
        )

        self.fitness = self.objective_function(
            self.chromosome
        )

    def evaluate(self):

        self.fitness = self.objective_function(
            self.chromosome
        )


# ============================================================
# Seleção por torneio
# ============================================================

def tournament_selection(
    population,
    tournament_size=5
):

    candidates = np.random.choice(
        population,
        tournament_size
    )

    best = min(
        candidates,
        key=lambda ind: ind.fitness
    )

    return best


# ============================================================
# Crossover aritmético
# ============================================================

def crossover(parent1, parent2):

    alpha = np.random.rand()

    child1 = alpha * parent1.chromosome + \
             (1 - alpha) * parent2.chromosome

    child2 = alpha * parent2.chromosome + \
             (1 - alpha) * parent1.chromosome

    return child1, child2


# ============================================================
# Mutação gaussiana
# ============================================================

def mutation(
    chromosome,
    bounds,
    mutation_rate=0.1,
    mutation_strength=2.0
):

    for i in range(len(chromosome)):

        if np.random.rand() < mutation_rate:

            chromosome[i] += np.random.normal(
                0,
                mutation_strength
            )

    chromosome = np.clip(
        chromosome,
        bounds[0],
        bounds[1]
    )

    return chromosome


# ============================================================
# Algoritmo Genético
# ============================================================

def genetic_algorithm(
    objective_function,
    dim=30,
    population_size=100,
    generations=1000,
    bounds=(-100, 100),
    crossover_rate=0.9,
    mutation_rate=0.1
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
    # Melhor indivíduo global
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

        new_population = []
        
        elite = min(
            population,
            key=lambda ind: ind.fitness
        )       

        while len(new_population) < population_size:

            # =================================================
            # Seleção
            # =================================================

            parent1 = tournament_selection(
                population,
            )

            parent2 = tournament_selection(
                population
            )

            # =================================================
            # Crossover
            # =================================================

            if np.random.rand() < crossover_rate:

                child1_chromosome, child2_chromosome = (
                    crossover(parent1, parent2)
                )

            else:

                child1_chromosome = (
                    parent1.chromosome.copy()
                )

                child2_chromosome = (
                    parent2.chromosome.copy()
                )

            # =================================================
            # Mutação
            # =================================================

            child1_chromosome = mutation(
                child1_chromosome,
                bounds,
                mutation_rate
            )

            child2_chromosome = mutation(
                child2_chromosome,
                bounds,
                mutation_rate
            )

            # =================================================
            # Cria filhos
            # =================================================

            child1 = Individual(
                dim,
                bounds,
                objective_function
            )

            child2 = Individual(
                dim,
                bounds,
                objective_function
            )

            child1.chromosome = child1_chromosome
            child2.chromosome = child2_chromosome

            child1.evaluate()
            child2.evaluate()

            new_population.append(child1)
            new_population.append(child2)

        #ELITISMO
        new_population[0] = elite
        
        # =====================================================
        # Atualiza população
        # =====================================================

        population = new_population[:population_size]

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
        best_individual.chromosome,
        best_individual.fitness,
        history
    )


# ============================================================
# Plot convergência
# ============================================================

def plot_convergence(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.title("Convergência do GA")

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
        genetic_algorithm(
            objective_function=objective_function,
            dim=30,
            population_size=50,
            generations=3000,
            bounds=bounds,
            crossover_rate=0.9,
            mutation_rate=0.1
        )
    )

    plot_convergence(history)

    print("\n===== RESULTADO FINAL =====")

    print("Melhor posição:")
    print(best_position)

    print("\nMelhor valor:")
    print(best_value)