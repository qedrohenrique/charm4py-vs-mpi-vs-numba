from numba import jit
import numpy as np
import time

MAX_M = 2048

@jit()
def mulmat(ma, mb, mc):
    for i in range(MAX_M):
        for j in range(MAX_M):
            for k in range(MAX_M):
                mc[i][j] += ma[i][k] * mb[k][j]


def main():
    for i in range(5):
        ma = np.random.randint(10, size=(MAX_M, MAX_M))
        mb = np.random.randint(10, size=(MAX_M, MAX_M))
        mc = np.zeros(shape=(MAX_M, MAX_M))

        start_time = time.time()
        mulmat(ma, mb, mc)
        end_time = (time.time() - start_time)

        print("[%dx%d]" % (MAX_M, MAX_M))
        print("Elapsed time: %.4f seconds" % end_time)
        print("%.4f\n" % end_time)


if __name__ == "__main__":
    main()
