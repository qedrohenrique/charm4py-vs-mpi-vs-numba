from numba import jit, prange, set_num_threads
import numpy as np
import time

MAX_ITER = 10_000
THRESHOLD = 0.0001
INITIAL_ERR = 1_000_000.0
MATRIX_SIZE = 512+2
THREADS = [1, 2, 4, 8, 16]


@jit(parallel=True, nogil=True, cache=False, nopython=True)
def compute_stencil(a, a_new, n_matrix):
    current_error = np.float64(0.0)
    for k in prange(2, (n_matrix - 2)):
        for j in range(2, (n_matrix - 2)):
            a_ith = (a[k][j + 1] + a[k][j - 1] + a[k - 1][j] + a[k + 1][j]) * 0.25
            current_error = max(current_error, abs(a_ith - a[k][j]))
            a_new[k][j] = a_ith

    return current_error


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


def main():

    for n_thread in THREADS:
        a, a_new = setup_matrixes(MATRIX_SIZE, MATRIX_SIZE)

        set_num_threads(n_thread)

        err = INITIAL_ERR
        iters = 0

        start_time = time.time()

        while err > THRESHOLD and iters < MAX_ITER:
            err = compute_stencil(a, a_new, MATRIX_SIZE)
            a, a_new = a_new, a
            iters += 1

        end_time = (time.time() - start_time)

        # import pandas as pd
        # print(pd.DataFrame(a))

        print("\n[%dx%d]" % (MATRIX_SIZE-2, MATRIX_SIZE-2))
        print("%d threads" % n_thread)
        print("%d iterations" % iters)
        print("Final error: ", err)
        print("Elapsed time: %.4f seconds" % end_time)
        print("%.4f\n" % end_time)


if __name__ == "__main__":
    main()
