from charm4py import charm, Chare, Array, Future, coro, Reducer
import numpy as np
import time
from numba import jit

MATRIX_SIZE = 2048


class MatrixMultiply(Chare):

    def __init__(self, ma, mb, sim_done_future):
        dimension = MATRIX_SIZE // charm.numPes()

        self.ma = ma
        self.mb = mb
        self.mc = np.zeros(shape=(dimension, MATRIX_SIZE))
        self.sim_done_future = sim_done_future

        self.start = charm.myPe() * dimension
        self.end = self.start + dimension - 1


    @coro
    def run(self):
        # print('PE [%d] - %d to %d' % (charm.myPe(), self.start, self.end))

        self.mc = compute(self.ma, self.mb, self.mc, self.start, self.end)

        self.reduce(self.sim_done_future, self.mc.tolist(), Reducer.gather)  # Sync all Chares


@jit(nopython=True, cache=False)
def compute(ma, mb, mc, start, end):
    for i in range(start, end + 1):
        for j in range(MATRIX_SIZE):
            for k in range(MATRIX_SIZE):
                mc_i = i - start
                mc[mc_i][j] += ma[i][k] * mb[k][j]

    return mc


def no_jit_compute(ma, mb, mc, start, end):
    for i in range(start, end + 1):
        for j in range(MATRIX_SIZE):
            for k in range(MATRIX_SIZE):
                mc_i = i - start
                mc[mc_i][j] += ma[i][k] * mb[k][j]


def main(args):
    ma = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))
    mb = np.random.randint(10, size=(MATRIX_SIZE, MATRIX_SIZE))

    sim_done = Future()

    array = Array(MatrixMultiply, charm.numPes(), args=[ma, mb, sim_done])
    charm.awaitCreation(array)

    initTime = time.time()
    array.run(awaitable=True)
    result = sim_done.get()
    totalTime = time.time() - initTime

    # flattened_result = []
    # for temp in result:
    #     for element in temp:
    #         flattened_result.append(element)
    #
    # print(flattened_result)

    print("[%dx%d]" % (MATRIX_SIZE, MATRIX_SIZE))
    print("%d processes" % (charm.numPes()))
    print("Elapsed time: %.4f seconds" % totalTime)
    print("%.4f\n" % totalTime)
    exit(0)

charm.start(main)

