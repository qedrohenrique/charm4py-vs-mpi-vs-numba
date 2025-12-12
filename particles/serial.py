# python serial.py 8 4 10000
# Versão serial do problema de simulação de partículas

import random
import math
import time
import sys

random.seed(42)

NUM_ITER = 100
SIM_BOX_SIZE = 100.0


class Particle:
    """Representa uma partícula no plano 2D."""

    def __init__(self, x: float, y: float):
        self.coords = [x, y]

    def perturb(self, cellsize):
        """Move a partícula aleatoriamente dentro de ±10% do tamanho da célula."""
        for i in range(2):
            self.coords[i] += random.uniform(-cellsize[i] * 0.1, cellsize[i] * 0.1)
            # Se a partícula sair dos limites, aparece do outro lado (wrap around)
            if self.coords[i] > SIM_BOX_SIZE:
                self.coords[i] -= SIM_BOX_SIZE
            elif self.coords[i] < 0:
                self.coords[i] += SIM_BOX_SIZE


def initial_num_particles(cell_coords, array_dims, max_particles, cellsize):
    """Define o número inicial de partículas de cada célula.

    Células mais próximas ao centro da grade começam com `max_particles`,
    as demais iniciam vazias, reproduzindo o desequilíbrio de carga do
    exemplo original.
    """
    grid_center = (SIM_BOX_SIZE / 2, SIM_BOX_SIZE / 2)
    cell_center = (
        cell_coords[0] * cellsize[0] + cellsize[0] / 2,
        cell_coords[1] * cellsize[1] + cellsize[1] / 2,
    )
    dist = math.hypot(cell_center[0] - grid_center[0], cell_center[1] - grid_center[1])
    return max_particles if dist <= SIM_BOX_SIZE / 5 else 0


def create_cell_array(dims_x, dims_y):
    """Cria estrutura 2-D de listas de partículas."""
    return [[[] for _ in range(dims_y)] for _ in range(dims_x)]


def main(argv):
    """Configura a simulação, cria partículas e executa a simulação serial."""
    
    # Parse argumentos
    if len(argv) >= 3:
        array_dims = (int(argv[1]), int(argv[2]))
    else:
        array_dims = (8, 4)  # default: 32 células

    if len(argv) >= 4:
        max_particles_per_cell_start = int(argv[3])
    else:
        max_particles_per_cell_start = 10000

    # Calcula tamanho de cada célula
    cellsize = (
        SIM_BOX_SIZE / array_dims[0],
        SIM_BOX_SIZE / array_dims[1],
    )

    # Cria a grade de células
    cells = create_cell_array(array_dims[0], array_dims[1])

    # Inicializa partículas em cada célula
    for cx in range(array_dims[0]):
        lo_x = cx * cellsize[0]
        for cy in range(array_dims[1]):
            lo_y = cy * cellsize[1]
            N = initial_num_particles((cx, cy), array_dims, max_particles_per_cell_start, cellsize)
            for _ in range(N):
                cells[cx][cy].append(
                    Particle(
                        random.uniform(lo_x, lo_x + cellsize[0] - 0.001),
                        random.uniform(lo_y, lo_y + cellsize[1] - 0.001),
                    )
                )

    # Conta partículas por célula
    particles_per_cell = [len(cells[cx][cy]) for cx in range(array_dims[0]) for cy in range(array_dims[1])]
    total_particles = sum(particles_per_cell)
    
    print("\n=== Simulação Serial de Partículas ===")
    print(f"Grade de células: {array_dims[0]} x {array_dims[1]}")
    print(f"Total de partículas: {total_particles}")
    print(f"Min / Max partículas por célula: {min(particles_per_cell)} / {max(particles_per_cell)}")
    print("\nIniciando simulação…")

    t0 = time.time()

    # Loop principal da simulação
    for it in range(NUM_ITER):
        # Lista temporária para partículas que precisam migrar
        particles_to_move = []

        # Processa cada célula
        for cx in range(array_dims[0]):
            for cy in range(array_dims[1]):
                cell_particles = cells[cx][cy]
                i = 0
                while i < len(cell_particles):
                    p = cell_particles[i]
                    p.perturb(cellsize)

                    # Calcula a célula de destino baseada na nova posição
                    dest_cx = int(p.coords[0] / cellsize[0]) % array_dims[0]
                    dest_cy = int(p.coords[1] / cellsize[1]) % array_dims[1]

                    if dest_cx == cx and dest_cy == cy:
                        # Partícula permanece na mesma célula
                        i += 1
                    else:
                        # Partícula precisa migrar para outra célula
                        particles_to_move.append((p, dest_cx, dest_cy))
                        # Remove da célula atual (swap com último elemento)
                        cell_particles[i] = cell_particles[-1]
                        cell_particles.pop()

        # Move partículas para suas novas células
        for p, dest_cx, dest_cy in particles_to_move:
            cells[dest_cx][dest_cy].append(p)

        # Reporta progresso a cada 10 iterações
        if it % 10 == 0:
            max_particles_in_cell = max(
                len(cells[cx][cy]) for cx in range(array_dims[0]) for cy in range(array_dims[1])
            )
            print(f"Iteração {it:3d}: máx partículas por célula = {max_particles_in_cell}")

    elapsed = time.time() - t0

    # Estatísticas finais
    final_particles_per_cell = [len(cells[cx][cy]) for cx in range(array_dims[0]) for cy in range(array_dims[1])]
    final_total = sum(final_particles_per_cell)
    
    print(f"\nSimulação concluída em {round(elapsed, 3)} segundos.")
    print(f"Total de partículas final: {final_total}")
    print(f"Min / Max partículas por célula final: {min(final_particles_per_cell)} / {max(final_particles_per_cell)}")


if __name__ == "__main__":
    main(sys.argv)

