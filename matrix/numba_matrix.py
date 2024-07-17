from numba import jit, prange, set_num_threads
import numpy as np
import time

MAX_M = 1600


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def mulmat(ma, mb, mc):
    for i in prange(MAX_M):
        for j in range(MAX_M):
            for k in range(MAX_M):
                mc[i][j] += ma[i][k] * mb[k][j]


def main():
    n_threads_array = [1, 2, 4, 8, 16]

    for n_thread in n_threads_array:
        set_num_threads(n_thread)

        ma = np.random.randint(10, size=(MAX_M, MAX_M))
        mb = np.random.randint(10, size=(MAX_M, MAX_M))
        mc = np.zeros(shape=(MAX_M, MAX_M))

        start_time = time.time()
        mulmat(ma, mb, mc)
        end_time = (time.time() - start_time)

        print("%d threads" % n_thread)
        print("Elapsed time: %.4f seconds\n" % end_time)


if __name__ == "__main__":
    main()
