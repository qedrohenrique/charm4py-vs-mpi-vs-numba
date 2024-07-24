from mpi4py import MPI
import numpy as np
from numba import jit
import sys
import time

MAX_ITER = 100_000
THRESHOLD = 0.0001
INITIAL_ERR = 1_000_000.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_processes = comm.Get_size()


def get_bounds():
    start = 1
    end = MATRIX_SIZE // num_processes

    if rank == 0:
        start = 2

    if rank == num_processes - 1:
        end = (MATRIX_SIZE // num_processes) - 1

    return int(start), int(end)


def jacobi(a, a_new):
    max_error = np.float64(0.0)
    iteration = 0
    converged = False

    start, end = get_bounds()

    while not converged and iteration < MAX_ITER:
        recv_buff_right = np.empty(MATRIX_SIZE - 2, dtype='f')
        recv_buff_left = np.empty(MATRIX_SIZE - 2, dtype='f')

        # Message to left
        send_data_left = np.array(a[2:MATRIX_SIZE, start], dtype='f')
        if rank != 0:  # Do not send message if you are a left boundary
            comm.Send(send_data_left, rank - 1, 20)

        if rank != num_processes - 1:
            comm.Recv(recv_buff_left, rank + 1, 20)

        # Message to right
        send_data_right = np.array(a[2:MATRIX_SIZE, end], dtype='f')
        if rank != num_processes - 1:  # Do not send message if you are a right boundary
            comm.Send(send_data_right, rank + 1, 10)

        if rank != 0:
            comm.Recv(recv_buff_right, rank - 1, 10)

        comm.Barrier()

        if rank != 0:
            a[2:MATRIX_SIZE, start - 1] = recv_buff_right  # Update left ghosts

        if rank != num_processes - 1:
            a[2:MATRIX_SIZE, end + 1] = recv_buff_left  # Update right ghosts

        max_error = check_and_compute(a, a_new, start, end)
        a, a_new = a_new, a
        converged = comm.allreduce(max_error < THRESHOLD, op=MPI.LAND)
        iteration += 1

    return iteration, max_error


@jit(nopython=True, cache=False)
def check_and_compute(a, a_new, start, end):
    max_error = np.float64(0.0)

    for i in range(2, MATRIX_SIZE):
        for j in range(start, end + 1):
            a_ith = (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]) * 0.25
            max_error = max(max_error, abs(a_ith - a[i, j]))
            a_new[i, j] = a_ith

    return max_error


def setup_input(args):
    global MATRIX_SIZE

    if len(args) != 2:
        if rank == 0:
            print('\nUsage:\t', args[0], 'array_size')
            exit()
        else:
            exit()

    MATRIX_SIZE = int(args[1])


def setup_matrixes(max_m, max_n):
    temperature = np.zeros((max_m, max_n), dtype=np.float64)
    new_temeperature = np.zeros((max_m, max_n), dtype=np.float64)

    # Top and Bottom constraints
    for i in range(0, max_n):
        temperature[1][i] = 1.0
        new_temeperature[1][i] = 1.0

        temperature[max_m - 2][i] = 1.0
        new_temeperature[max_m - 2][i] = 1.0

    # Left constraint
    if rank == 0:
        for i in range(0, max_m):
            temperature[i][1] = 1.0
            new_temeperature[i][1] = 1.0

    # Right constraint
    if rank == num_processes - 1:
        for i in range(0, max_m):
            temperature[i][max_n - 2] = 1.0
            new_temeperature[i][max_n - 2] = 1.0

    return temperature, new_temeperature


def main(args):
    setup_input(args)

    a, a_new = setup_matrixes(MATRIX_SIZE + 2, (MATRIX_SIZE//num_processes) + 2)

    initTime = time.time()
    its, err = jacobi(a, a_new)
    totalTime = time.time() - initTime

    if rank == 0:
        print("\n[%dx%d]" % (MATRIX_SIZE, MATRIX_SIZE ))
        print("%d processes" % num_processes)
        print("%d iterations" % its)
        print("Final error: ", err)
        print("Elapsed time: %.4f seconds" % totalTime)
        print("%.4f\n" % totalTime)


if __name__ == '__main__':
    main(sys.argv)
