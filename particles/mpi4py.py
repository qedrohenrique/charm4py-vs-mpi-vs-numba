#  mpiexec -n 4 python -u mpi.py 8 4 10000
#  python3 -m charmrun.start +p4 particles.py 8 4 10000 +balancer GreedyRefineLB

# Em linhas gerais, mpi é mais rapido que o charm.
# Mas ao acenturamos o desbalanceamento de carga, o charm se torna mais rapido, quando usamos o balances
# Para desbalancear a carga, queremos processos livres e procesos ocupados, então podemos:
# - Aumentar o numero de processos (mais processos livres) -> Parece ter o maior impacto
# - Aumentar o numero de partículas por célula (maior trabalho por processo)

from mpi4py import MPI
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
            if self.coords[i] > SIM_BOX_SIZE:
                self.coords[i] -= SIM_BOX_SIZE
            elif self.coords[i] < 0:
                self.coords[i] += SIM_BOX_SIZE


def initial_num_particles(cell_coords, array_dims, max_particles, cellsize):
    """Define o número inicial de partículas de cada célula.

    Células mais próximas ao centro da grade começam com `max_particles`,
    as demais iniciam vazias, reproduzindo o desequilíbrio de carga do
    exemplo original em Charm4py.
    """
    grid_center = (SIM_BOX_SIZE / 2, SIM_BOX_SIZE / 2)
    cell_center = (
        cell_coords[0] * cellsize[0] + cellsize[0] / 2,
        cell_coords[1] * cellsize[1] + cellsize[1] / 2,
    )
    dist = math.hypot(cell_center[0] - grid_center[0], cell_center[1] - grid_center[1])
    return max_particles if dist <= SIM_BOX_SIZE / 5 else 0

def compute_proc_grid(size):
    """Devolve (px, py) tal que px*py == size e ambos > 0.

    Tenta usar MPI.Compute_dims; se alguma dimensão vier 0, recorre a fatoração
    baseada em raiz quadrada para manter grade quase quadrada.
    """
    try:
        px, py = MPI.Compute_dims(size, 2)
    except TypeError:
        # versões mais antigas de mpi4py usam assinatura (nnodes, dims_list)
        dims = [0, 0]
        MPI.Compute_dims(size, dims)
        px, py = dims

    # Se ainda veio algo inválido, calcula manualmente
    if px == 0 or py == 0:
        px = int(math.sqrt(size))
        while size % px != 0:
            px -= 1
        py = size // px
    return (px, py)


def owner_rank(cart_comm, global_cx, global_cy, local_cells_x, local_cells_y):
    """Dado índice global de célula, devolve rank MPI dono da célula."""
    proc_coords = (
        (global_cx // local_cells_x),
        (global_cy // local_cells_y),
    )
    return cart_comm.Get_cart_rank(proc_coords)


def create_local_cell_array(local_x, local_y):
    """Cria estrutura 2-D de listas de partículas."""
    return [[[] for _ in range(local_y)] for _ in range(local_x)]


def main(argv):
    """Configura MPI, cria partículas (com tileamento) e roda simulação."""
    if len(argv) >= 3:
        array_dims = (int(argv[1]), int(argv[2]))
    else:
        array_dims = (8, 4)  # default 32 células

    if len(argv) >= 4:
        max_particles_per_cell_start = int(argv[3])
    else:
        max_particles_per_cell_start = 10000

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    proc_dims = compute_proc_grid(size)
    if proc_dims[0] * proc_dims[1] != size:
        if rank == 0:
            print("Não foi possível encontrar grade cartesiana 2-D para", size, "processos.")
        sys.exit(1)

    if array_dims[0] % proc_dims[0] != 0 or array_dims[1] % proc_dims[1] != 0:
        if rank == 0:
            print(
                f"Erro: dimensões globais de células {array_dims} não são divisíveis pela grade de processos {proc_dims}."
            )
        sys.exit(1)

    local_cells_x = array_dims[0] // proc_dims[0]
    local_cells_y = array_dims[1] // proc_dims[1]

    cart_comm = comm.Create_cart(dims=proc_dims, periods=(True, True), reorder=False)
    proc_coords = cart_comm.Get_coords(rank)

    cellsize = (
        SIM_BOX_SIZE / array_dims[0],
        SIM_BOX_SIZE / array_dims[1],
    )

    start_cx = proc_coords[0] * local_cells_x
    start_cy = proc_coords[1] * local_cells_y

    cells = create_local_cell_array(local_cells_x, local_cells_y)

    for lx in range(local_cells_x):
        global_cx = start_cx + lx
        lo_x = global_cx * cellsize[0]
        for ly in range(local_cells_y):
            global_cy = start_cy + ly
            lo_y = global_cy * cellsize[1]
            N = initial_num_particles((global_cx, global_cy), array_dims, max_particles_per_cell_start, cellsize)
            for _ in range(N):
                cells[lx][ly].append(
                    Particle(
                        random.uniform(lo_x, lo_x + cellsize[0] - 0.001),
                        random.uniform(lo_y, lo_y + cellsize[1] - 0.001),
                    )
                )

    local_count = sum(len(cells[lx][ly]) for lx in range(local_cells_x) for ly in range(local_cells_y))
    counts = comm.gather(local_count, root=0)
    if rank == 0:
        print("\nGrade de células global:", array_dims[0], "x", array_dims[1])
        print("Grade de processos:", proc_dims[0], "x", proc_dims[1])
        print("Total partículas:", sum(counts))
        print("Min / Max partículas por processo:", min(counts), "/", max(counts))
        print("\nIniciando simulação…")
    comm.Barrier()
    t0 = time.time()

    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    neighbor_ranks = list(
        {
            cart_comm.Get_cart_rank(
                ((proc_coords[0] + dx) % proc_dims[0], (proc_coords[1] + dy) % proc_dims[1])
            )
            for dx, dy in neighbor_offsets
            if not (dx == 0 and dy == 0)
        }
    )

    for it in range(NUM_ITER):
        outgoing = {nb: [] for nb in neighbor_ranks}

        for lx in range(local_cells_x):
            for ly in range(local_cells_y):
                cell_particles = cells[lx][ly]
                i = 0
                while i < len(cell_particles):
                    p = cell_particles[i]
                    p.perturb(cellsize)

                    dest_cx = int(p.coords[0] / cellsize[0]) % array_dims[0]
                    dest_cy = int(p.coords[1] / cellsize[1]) % array_dims[1]

                    dest_rank = owner_rank(cart_comm, dest_cx, dest_cy, local_cells_x, local_cells_y)

                    if dest_rank == rank:
                        dest_lx = dest_cx - start_cx
                        dest_ly = dest_cy - start_cy
                        if dest_lx == lx and dest_ly == ly:
                            i += 1 
                        else:
                            cell_particles[i] = cell_particles[-1]
                            cell_particles.pop()
                            cells[dest_lx][dest_ly].append(p)
                    else:
                        outgoing.setdefault(dest_rank, []).append(p.coords)
                        cell_particles[i] = cell_particles[-1]
                        cell_particles.pop()

        send_reqs = [cart_comm.isend(outgoing.get(nb, []), dest=nb, tag=it) for nb in neighbor_ranks]

        incoming_particles = []
        for nb in neighbor_ranks:
            recv_data = cart_comm.recv(source=nb, tag=it)
            for xy in recv_data:
                incoming_particles.append(Particle(xy[0], xy[1]))

        for req in send_reqs:
            req.wait()

        for p in incoming_particles:
            dest_cx = int(p.coords[0] / cellsize[0]) % array_dims[0]
            dest_cy = int(p.coords[1] / cellsize[1]) % array_dims[1]
            dest_lx = dest_cx - start_cx
            dest_ly = dest_cy - start_cy
            cells[dest_lx][dest_ly].append(p)

        if it % 10 == 0:
            local_max_cell = 0
            for lx in range(local_cells_x):
                for ly in range(local_cells_y):
                    local_max_cell = max(local_max_cell, len(cells[lx][ly]))

            global_max_cell = comm.allreduce(local_max_cell, op=MPI.MAX)

            if rank == 0:
                print(
                    f"Iteração {it:3d}: máx partículas por célula = {global_max_cell}"
                )

        if it % 20 == 0:
            comm.Barrier()

    comm.Barrier()
    elapsed = time.time() - t0
    if rank == 0:
        print("\nSimulação concluída em", round(elapsed, 3), "segundos.")

if __name__ == "__main__":
    main(sys.argv)