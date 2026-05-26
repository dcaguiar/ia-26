import numpy as np
import math


# ============================================================
# Classe Nó
# ============================================================

class Node:

    def __init__(self, position):

        self.position = position

        # ================================================
        # Custos
        # ================================================

        self.g = np.inf

        self.h = 0

        self.f = np.inf

        # ================================================
        # Pai
        # ================================================

        self.parent = None


# ============================================================
# Heurística
# Distância Euclidiana
# ============================================================

def heuristic(current, goal):

    return math.sqrt(
        (current[0] - goal[0])**2
        +
        (current[1] - goal[1])**2
    )


# ============================================================
# Reconstrução do caminho
# ============================================================

def reconstruct_path(node):

    path = []

    current = node

    while current is not None:

        path.append(current.position)

        current = current.parent

    return path[::-1]


# ============================================================
# A*
# ============================================================

def a_star(grid, start, goal):

    rows = len(grid)

    cols = len(grid[0])

    # ========================================================
    # Cria matriz de nós
    # ========================================================

    nodes = [

        [Node((i, j)) for j in range(cols)]

        for i in range(rows)
    ]

    # ========================================================
    # Nó inicial
    # ========================================================

    start_node = nodes[start[0]][start[1]]

    start_node.g = 0

    start_node.h = heuristic(
        start,
        goal
    )

    start_node.f = (
        start_node.g
        + start_node.h
    )

    # ========================================================
    # Listas
    # ========================================================

    open_list = [start_node]

    closed_list = []

    # ========================================================
    # Movimentos
    # ========================================================

    directions = [

        (-1, 0),  # cima
        (1, 0),   # baixo
        (0, -1),  # esquerda
        (0, 1)    # direita
    ]

    # ========================================================
    # Enquanto houver abertas
    # ========================================================

    while len(open_list) > 0:

        # ====================================================
        # Remove menor F
        # ====================================================

        current_node = min(
            open_list,
            key=lambda node: node.f
        )

        open_list.remove(current_node)

        closed_list.append(
            current_node.position
        )

        # ====================================================
        # Destino
        # ====================================================

        if current_node.position == goal:

            return reconstruct_path(
                current_node
            ), current_node.g

        # ====================================================
        # Expansão
        # ====================================================

        for direction in directions:

            row = (
                current_node.position[0]
                + direction[0]
            )

            col = (
                current_node.position[1]
                + direction[1]
            )

            # =================================================
            # Limites
            # =================================================

            if (
                row < 0
                or row >= rows
                or col < 0
                or col >= cols
            ):

                continue

            # =================================================
            # Obstáculo
            # =================================================

            if grid[row][col] == "*":

                continue

            # =================================================
            # Fechada
            # =================================================

            if (row, col) in closed_list:

                continue

            neighbor = nodes[row][col]

            # =================================================
            # Custo da ação
            # =================================================

            cell_cost = grid[row][col]

            # =================================================
            # G = Ganterior + custo
            # =================================================

            g_new = (
                current_node.g
                + cell_cost
            )

            # =================================================
            # Heurística
            # =================================================

            h_new = heuristic(
                (row, col),
                goal
            )

            # =================================================
            # f = g + h
            # =================================================

            f_new = g_new + h_new

            # =================================================
            # Atualiza se melhor
            # =================================================

            if f_new < neighbor.f:

                neighbor.g = g_new

                neighbor.h = h_new

                neighbor.f = f_new

                neighbor.parent = current_node

                if neighbor not in open_list:

                    open_list.append(
                        neighbor
                    )

    return None, np.inf



# Exemplo 

if __name__ == "__main__":

    grid = [

        ["I", 2, 1, 1, 1, 2],

        [1, "*", 2, "*", 3, 1],

        [1, 4, 15, "*", "*", 1],

        [2, "*", 15, "*", 4, 1],

        ["*", "*", 2, 2, 9, 1],

        [2, 1, 1, 1, "F", 1]
    ]

    # ========================================================
    # Conversão
    # ========================================================

    start = None

    goal = None

    for i in range(len(grid)):

        for j in range(len(grid[0])):

            if grid[i][j] == "I":

                start = (i, j)

                grid[i][j] = 0

            elif grid[i][j] == "F":

                goal = (i, j)

                grid[i][j] = 0

    # ========================================================
    # Executa
    # ========================================================

    path, cost = a_star(
        grid,
        start,
        goal
    )

    # ========================================================
    # Resultado
    # ========================================================

    print("\n===== CAMINHO =====\n")

    for p in path:

        print(p)

    print("\nCusto total:")

    print(cost)