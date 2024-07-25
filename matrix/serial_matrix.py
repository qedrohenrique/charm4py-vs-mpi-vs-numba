from numba import jit
import numpy as np
import time

MATRIX_SIZE = 2048


@jit(nopython=True, cache=False)
def mulmat(ma, mb, mc):
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            for k in range(MATRIX_SIZE):
                mc[i][j] += ma[i][k] * mb[k][j]


def main():
    for i in range(5):
        ma = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
        mb = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
        mc = np.zeros(shape=(MATRIX_SIZE, MATRIX_SIZE))

        start_time = time.time()
        mulmat(ma, mb, mc)
        end_time = (time.time() - start_time)

        print("[%dx%d]" % (MATRIX_SIZE, MATRIX_SIZE))
        print("Elapsed time: %.4f seconds" % end_time)
        print("%.4f\n" % end_time)


if __name__ == "__main__":
    main()
