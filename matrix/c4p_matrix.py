from charm4py import charm, coro
import numpy as np
import time


MATRIX_SIZE = 200


@coro
def matrix_multiply(data):
    ma, mb, i, j = data
    result = 0
    for k in range(MATRIX_SIZE):
        result += ma[i][k] * mb[k][j]
    return result


def main(args):
    ma = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
    mb = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))

    tasks = [(ma, mb, i, j) for i in range(MATRIX_SIZE) for j in range(MATRIX_SIZE)]

    start_time = time.time()
    result = charm.pool.map(
        matrix_multiply,
        tasks
    )
    end_time = (time.time() - start_time)

    print("Elapsed time: %.4f seconds\n" % end_time)

    # matrix_result = np.array(result).reshape(MATRIX_SIZE, MATRIX_SIZE)
    # print(matrix_result)
    exit(0)


charm.start(main)
