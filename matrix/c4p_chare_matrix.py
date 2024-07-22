from charm4py import charm, Chare, Array, Future, coro, Reducer
import numpy as np
import time

MAX_M = 512

class MatrixMultiply(Chare):

    def __init__(self, ma, mb, sim_done_future):
        dimension = MAX_M // charm.numPes()

        self.ma = ma
        self.mb = mb
        self.mc = np.zeros(shape=(dimension, MAX_M))
        self.sim_done_future = sim_done_future

        self.start = charm.myPe() * dimension
        self.end = self.start + dimension - 1

        self.done = False

    @coro
    def run(self):
        # print('PE [%d] - %d to %d' % (charm.myPe(), self.start, self.end))

        for i in range(self.start, self.end + 1):
            for j in range(MAX_M):
                for k in range(MAX_M):
                    mc_i = i - self.start
                    self.mc[mc_i][j] += self.ma[i][k] * self.mb[k][j]

        self.reduce(self.sim_done_future, self.mc.tolist(), Reducer.gather)  # Sync all Chares


def main(args):
    ma = np.random.randint(10, size=(MAX_M, MAX_M))
    mb = np.random.randint(10, size=(MAX_M, MAX_M))

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

    print("[%dx%d]" % (MAX_M, MAX_M))
    print("%d processes" % (charm.numPes()))
    print("Elapsed time: %.4f seconds" % totalTime)
    print("%.4f\n" % totalTime)

charm.start(main)

