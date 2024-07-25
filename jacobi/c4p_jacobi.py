from charm4py import charm, Chare, Group, Array, Future, coro, Channel, Reducer
import time
import numpy as np

try:
    from numba import jit

    numbaFound = True
except ImportError:
    numbaFound = False

    def jit(*args, **kwargs):
        def deco(func):
            return func

        return deco

MAX_ITER = 10_000
LEFT, RIGHT, TOP, BOTTOM = range(4)
THRESHOLD = 0.0001
INITIAL_ERR = 1_000_000.0


class Jacobi(Chare):
    def __init__(self, sim_done_future):
        self.sim_done_future = sim_done_future
        self.temperature = np.zeros((blockDimX + 2, blockDimY + 2), dtype=np.float64)
        self.new_temperature = np.zeros((blockDimX + 2, blockDimY + 2), dtype=np.float64)
        self.leftBound = self.rightBound = self.topBound = self.bottomBound = False
        self.istart = self.jstart = 1
        self.ifinish = blockDimX + 1
        self.jfinish = blockDimY + 1
        self.nbs = []
        x, y = self.thisIndex

        if x == 0:
            self.leftBound = True
            self.istart += 1
        else:
            self.left_nb = Channel(self, remote=self.thisProxy[(x - 1, y)])
            self.nbs.append(self.left_nb)

        if x == num_chare_x - 1:
            self.rightBound = True
            self.ifinish -= 1
        else:
            self.right_nb = Channel(self, remote=self.thisProxy[(x + 1, y)])
            self.nbs.append(self.right_nb)

        if y == 0:
            self.topBound = True
            self.jstart += 1
        else:
            self.top_nb = Channel(self, remote=self.thisProxy[(x, y - 1)])
            self.nbs.append(self.top_nb)

        if y == num_chare_y - 1:
            self.bottomBound = True
            self.jfinish -= 1
        else:
            self.bottom_nb = Channel(self, remote=self.thisProxy[(x, y + 1)])
            self.nbs.append(self.bottom_nb)

        self.constrainBC()


    @coro
    def run(self):
        """ this is the main computation loop """
        iteration = 0
        max_error = INITIAL_ERR
        converged = False
        while not converged and iteration < MAX_ITER:
            if not self.leftBound:
                self.left_nb.send(RIGHT, self.temperature[1, 1:blockDimY + 1])
            if not self.rightBound:
                self.right_nb.send(LEFT, self.temperature[blockDimX, 1:blockDimY + 1])
            if not self.topBound:
                self.top_nb.send(BOTTOM, self.temperature[1:blockDimX + 1, 1])
            if not self.bottomBound:
                self.bottom_nb.send(TOP, self.temperature[1:blockDimX + 1, blockDimY])

            for nb in charm.iwait(self.nbs):
                direction, ghosts = nb.recv()
                if direction == LEFT:
                    self.temperature[0, 1:len(ghosts) + 1] = ghosts
                elif direction == RIGHT:
                    self.temperature[blockDimX + 1, 1:len(ghosts) + 1] = ghosts
                elif direction == TOP:
                    self.temperature[1:len(ghosts) + 1, 0] = ghosts
                elif direction == BOTTOM:
                    self.temperature[1:len(ghosts) + 1, blockDimY + 1] = ghosts
                else:
                    charm.abort('Invalid direction')

            max_error = check_and_compute(self.temperature, self.new_temperature,
                                          self.istart, self.ifinish, self.jstart, self.jfinish)
            self.temperature, self.new_temperature = self.new_temperature, self.temperature
            converged = self.allreduce(max_error <= THRESHOLD, Reducer.logical_and).get()
            iteration += 1

        if self.thisIndex == (0, 0):
            self.sim_done_future.send([iteration, max_error])

    def constrainBC(self):
        if self.leftBound:
            self.temperature[0:blockDimX + 2, 1] = 1.0
            self.new_temperature[0:blockDimX + 2, 1] = 1.0
        if self.topBound:
            self.temperature[1, 0:blockDimY + 2] = 1.0
            self.new_temperature[1, 0:blockDimY + 2] = 1.0
        if self.rightBound:
            self.temperature[0:blockDimX + 2, blockDimY] = 1.0
            self.new_temperature[0:blockDimX + 2, blockDimY] = 1.0
        if self.bottomBound:
            self.temperature[blockDimX, 0:blockDimY + 2] = 1.0
            self.new_temperature[blockDimX, 0:blockDimY + 2] = 1.0


@jit(nopython=True, cache=False)
def check_and_compute(temperature, new_temperature, istart, ifinish, jstart, jfinish):
    max_error = np.float64(0.0)
    for i in range(istart, ifinish):
        for j in range(jstart, jfinish):
            temperature_ith = (temperature[i - 1, j] + temperature[i + 1, j]
                               + temperature[i, j - 1] + temperature[i, j + 1]) * 0.25
            max_error = max(max_error, abs(temperature_ith - temperature[i, j]))
            new_temperature[i, j] = temperature_ith

    return max_error


class Util(Chare):
    def compile(self):
        T = np.zeros((blockDimX + 2, blockDimY + 2), dtype=np.float64)
        NT = np.zeros((blockDimX + 2, blockDimY + 2), dtype=np.float64)
        check_and_compute(T, NT, 1, blockDimX + 1, 1, blockDimY + 1)


def main(args):
    global blockDimX, blockDimY, num_chare_x, num_chare_y
    if len(args) != 3 and len(args) != 5:
        print('\nUsage:\t', args[0], 'array_size block_size')
        print('\t', args[0], 'array_size_X array_size_Y block_size_X block_size_Y')
        exit()

    if len(args) == 3:
        arrayDimX = arrayDimY = int(args[1])
        blockDimX = blockDimY = int(args[2])
    elif len(args) == 5:
        arrayDimX, arrayDimY = [int(arg) for arg in args[1:3]]
        blockDimX, blockDimY = [int(arg) for arg in args[3:5]]

    assert (arrayDimX >= blockDimX) and (arrayDimX % blockDimX == 0)
    assert (arrayDimY >= blockDimY) and (arrayDimY % blockDimY == 0)
    num_chare_x = arrayDimX // blockDimX
    num_chare_y = arrayDimY // blockDimY

    charm.thisProxy.updateGlobals({'blockDimX': blockDimX,
                                   'blockDimY': blockDimY,
                                   'num_chare_x': num_chare_x,
                                   'num_chare_y': num_chare_y},
                                  awaitable=True).get()

    print('\nRunning Jacobi on', charm.numPes(), 'processors with', num_chare_x, 'x', num_chare_y, 'chares')
    print('Array Dimensions:', arrayDimX, 'x', arrayDimY)
    print('Block Dimensions:', blockDimX, 'x', blockDimY)
    print('Max iterations:', MAX_ITER)
    print('Threshold:', THRESHOLD)

    if numbaFound:
        Group(Util).compile(awaitable=True).get()
        print('Numba compilation complete')
    else:
        print('!!WARNING!! Numba not found. Will run without Numba but it will be very slow')

    sim_done = Future()
    array = Array(Jacobi, (num_chare_x, num_chare_y), args=[sim_done])
    charm.awaitCreation(array)

    print('Starting computation')
    initTime = time.time()
    array.run()
    total_iterations, max_err = sim_done.get()
    totalTime = time.time() - initTime

    print("\n[%dx%d]" % (arrayDimX, arrayDimX))
    print("%d processes" % charm.numPes())
    print("%d iterations" % total_iterations)
    print("Final error: ", max_err)
    print("Elapsed time: %.4f seconds" % totalTime)
    print("%.4f\n" % totalTime)

    exit()

charm.start(main)
