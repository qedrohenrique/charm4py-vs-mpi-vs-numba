from charm4py import charm, coro
import numpy as np
import time


MATRIX_SIZE = 512


def time_limit(i):
    return (time.time() - start_time) > 300 and i % 10 == 0


@coro
def matrix_multiply(data):
    ma, mb, i, j = data
    result = 0

    if time_limit(i):
        print("[%dx%d]" % (MATRIX_SIZE, MATRIX_SIZE))
        print("%d processes" % charm.numPes())
        print("Time limit of 300 seconds exceedeed")
        exit(0)

    for k in range(MATRIX_SIZE):
        result += ma[i][k] * mb[k][j]
    return result


def main(args):
    global start_time

    ma = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
    mb = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))

    tasks = [(ma, mb, i, j) for i in range(MATRIX_SIZE) for j in range(MATRIX_SIZE)]

    start_time = time.time()
    charm.thisProxy.updateGlobals({'start_time': start_time}, awaitable=True).get()
    result = charm.pool.map(
        matrix_multiply,
        tasks
    )
    end_time = (time.time() - start_time)

    print("[%dx%d]" % (MATRIX_SIZE, MATRIX_SIZE))
    print("%d processes" % charm.numPes())
    print("Elapsed time: %.4f seconds" % end_time)
    print("%.4f\n" % end_time)

    # matrix_result = np.array(result).reshape(MATRIX_SIZE, MATRIX_SIZE)
    # print(matrix_result)
    exit(0)


charm.start(main)
