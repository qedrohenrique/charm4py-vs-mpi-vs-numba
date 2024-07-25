import numpy as np
from mpi4py import MPI
import time
from numba import jit


MATRIX_SIZE = 2048

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processes = comm.Get_size()
dimension = MATRIX_SIZE // num_processes


def matrix_multiply(ma, mb, mc):
    mc = compute(ma, mb, mc)

    if rank == 0:
        full_matrix = np.empty((MATRIX_SIZE, MATRIX_SIZE), dtype='i')
    else:
        full_matrix = None

    comm.Gather(mc, full_matrix, root=0)

    return full_matrix


@jit(nopython=True, cache=False)
def compute(ma, mb, mc):

    for i in range(0, dimension):
        for j in range(MATRIX_SIZE):
            for k in range(MATRIX_SIZE):
                mc[i][j] += ma[i][k] * mb[k][j]

    return mc


def main():
    ma = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
    mb = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
    mc = np.zeros(shape=(dimension, MATRIX_SIZE), dtype='i')

    initTime = time.time()
    result = matrix_multiply(ma, mb, mc)
    totalTime = time.time() - initTime

    if rank == 0:
        print("[%dx%d]" % (MATRIX_SIZE, MATRIX_SIZE))
        print("%d processes" % num_processes)
        print("Elapsed time: %.4f seconds" % totalTime)
        print("%.4f\n" % totalTime)


if __name__ == "__main__":
    main()
