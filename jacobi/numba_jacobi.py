from numba import jit, prange, set_num_threads
import numpy as np
import time

MAX_ITER = 100000
MAX_TOL = 0.0001
INITIAL_ERR = 1000000.0
THREADS = [1, 2, 4, 8, 16]
MATRIX_SIZES = [2056+2]


@jit(parallel=True, nogil=True, cache=False, nopython=True)
def compute_stencil(a, a_new, n_matrix):
    current_error = np.float64(0.0)
    for k in prange(2, (n_matrix - 2)):
        for j in range(2, (n_matrix - 2)):
            a_ith = (a[k][j] + a[k][j + 1] + a[k][j - 1] + a[k - 1][j] + a[k + 1][j]) * 0.2
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
        set_num_threads(n_thread)

        for n_matrix in MATRIX_SIZES:
            err = INITIAL_ERR
            iters = 0
            a, a_new = setup_matrixes(n_matrix, n_matrix)

            start_time = time.time()

            while err > MAX_TOL and iters < MAX_ITER:
                err = compute_stencil(a, a_new, n_matrix)
                a, a_new = a_new, a
                iters += 1

            end_time = (time.time() - start_time)

            # import pandas as pd
            # print(pd.DataFrame(a))

            print("\n[%dx%d]" % (n_matrix-2, n_matrix-2))
            print("%d threads" % n_thread)
            print("%d iterations" % iters)
            print("Final error: ", err)
            print("Elapsed time: %.4f seconds" % end_time)
            print("%.4f\n" % end_time)


if __name__ == "__main__":
    main()
