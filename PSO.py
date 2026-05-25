import numpy as np
import matplotlib.pyplot as plt

import functions as f


# ============================================================
# Classe Partícula
# ============================================================

class Particle:

    def __init__(self, dim, bounds, objective_function):

        self.objective_function = objective_function

        self.position = np.random.uniform(
            bounds[0],
            bounds[1],
            dim
        )

        self.velocity = np.random.uniform(
            -1,
            1,
            dim
        )

        self.best_position = self.position.copy()

        self.best_value = self.objective_function(
            self.position
        )

    def update_velocity(
        self,
        global_best,
        w,
        c1,
        c2
    ):

        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))

        cognitive = (
            c1 * r1 *
            (self.best_position - self.position)
        )

        social = (
            c2 * r2 *
            (global_best - self.position)
        )

        self.velocity = (
            w * self.velocity
            + cognitive
            + social
        )

    def update_position(self, bounds):

        self.position += self.velocity

        self.position = np.clip(
            self.position,
            bounds[0],
            bounds[1]
        )

        value = self.objective_function(
            self.position
        )

        if value < self.best_value:

            self.best_value = value

            self.best_position = (
                self.position.copy()
            )


# ============================================================
# PSO
# ============================================================

def particle_swarm_optimization(
    objective_function,
    dim=2,
    n_particles=50,
    max_iter=200,
    bounds=(-100, 100),
    w=0.7,
    c1=1.5,
    c2=1.5
):

    swarm = [
        Particle(
            dim,
            bounds,
            objective_function
        )
        for _ in range(n_particles)
    ]

    global_best_position = (
        swarm[0].best_position.copy()
    )

    global_best_value = (
        swarm[0].best_value
    )

    history = []

    for particle in swarm:

        if particle.best_value < global_best_value:

            global_best_value = (
                particle.best_value
            )

            global_best_position = (
                particle.best_position.copy()
            )

    for iteration in range(max_iter):

        for particle in swarm:

            particle.update_velocity(
                global_best_position,
                w,
                c1,
                c2
            )

            particle.update_position(bounds)

            if particle.best_value < global_best_value:

                global_best_value = (
                    particle.best_value
                )

                global_best_position = (
                    particle.best_position.copy()
                )

            print(
            f"Iteração {iteration+1:5d}"
            f" | Melhor valor: "
            f"{global_best_value:.10f}"
        )

        history.append(global_best_value)


    return (
        global_best_position,
        global_best_value,
        history
    )


# ============================================================
# Plot convergência
# ============================================================

def plot_convergence(history):

    plt.figure(figsize=(8, 5))

    plt.plot(history)

    plt.title("Convergência do PSO")

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

    objective_function = f.f1

    bounds = (-100, 100)

    # ========================================================
    # 2) Executa PSO
    # ========================================================

    best_position, best_value, history = (
        particle_swarm_optimization(
            objective_function=objective_function,
            dim=30,
            n_particles=100,
            max_iter=1000,
            bounds=bounds
        )
    )

    # ========================================================
    # 3) Plot convergência
    # ========================================================

    plot_convergence(history)

    print("\n===== RESULTADO FINAL =====")

    print("Melhor posição:")
    print(best_position)

    print("\nMelhor valor:")
    print(best_value)