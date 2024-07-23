from mpi4py import MPI
import numpy as np
import sys

MAX_ITER = 10000
THRESHOLD = 0.0001
INITIAL_ERR = 1_000_000.0
MATRIX_SIZE = 512+2


def jacobi(a, a_new):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        a[0][0] = 12
        comm.send(a[0][0], 1, 10)
    if rank == 1:
        data = comm.recv(source=0, tag=10)
        print(data)
        print(a[0][0])

    exit()


def setup_input(args):
    global arrayDimX, arrayDimY, blockDimX, blockDimY, num_chare_x, num_chare_y

    if len(args) != 3 and len(args) != 5:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('\nUsage:\t', args[0], 'array_size block_size')
            print('\t', args[0], 'array_size_X array_size_Y block_size_X block_size_Y')
            exit()
        else:
            exit()

    if len(args) == 3:
        arrayDimX = arrayDimY = int(args[1])
        blockDimX = blockDimY = int(args[2])
    elif len(args) == 5:
        arrayDimX, arrayDimY = [int(arg) for arg in args[1:3]]
        blockDimX, blockDimY = [int(arg) for arg in args[3:5]]

    assert (arrayDimX >= blockDimX) and (arrayDimX % blockDimX == 0)
    assert (arrayDimY >= blockDimY) and (arrayDimY % blockDimY == 0)

    num_chare_x = arrayDimX // blockDimX
    num_chare_y = arrayDimY // blockDimY


def setup_matrixes(max_m, max_n):
    temperature = np.zeros((max_m, max_n), dtype=np.float64)
    new_temeperature = np.zeros((max_m, max_n), dtype=np.float64)

    for i in range(0, max_m):
        temperature[1][i] = 1.0
        new_temeperature[1][i] = 1.0

    for i in range(0, max_m):
        temperature[max_m - 2][i] = 1.0
        new_temeperature[max_m - 2][i] = 1.0

    for i in range(0, max_m):
        temperature[i][1] = 1.0
        new_temeperature[i][1] = 1.0

    for i in range(0, max_m):
        temperature[i][max_m - 2] = 1.0
        new_temeperature[i][max_m - 2] = 1.0

    return temperature, new_temeperature


def main(args):
    setup_input(args)

    a, a_new = setup_matrixes(arrayDimX, arrayDimY)

    jacobi(a, a_new)


if __name__ == '__main__':
    main(sys.argv)
