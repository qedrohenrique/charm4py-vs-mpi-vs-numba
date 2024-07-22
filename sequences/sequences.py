import time

import numba
from charm4py import coro, charm, Chare, Group
import click

#
# Fibonacci
#

@coro
def fibonacci(n):
    if n < GRAINSIZE:
        return fib_seq(n)
    else:
        return sum(charm.pool.map(fibonacci, [n - 1, n - 2]))


@numba.jit(nopython=True, cache=False)
def fib_seq(n):
    if n < 2:
        return n
    else:
        return fib_seq(n-1) + fib_seq(n-2)


class FibUtil(Chare):
    def compile(self):
        fib_seq(3)


#
# Lucas
#

@coro
def lucas(n):
    if n <= GRAINSIZE:
        return lucas_seq(n)
    else:
        return sum(charm.pool.map(lucas, [n - 1, n - 2]))


@numba.jit(nopython=True, cache=False)  # numba really speeds up the computation
def lucas_seq(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1
    else:
        return lucas_seq(n-1) + lucas_seq(n-2)


class LucasUtil(Chare):
    def compile(self):
        lucas_seq(3)


@coro
@click.command()
@click.option('--sequence',
              default='Fibonacci',
              help='Sequence used.',
              type=click.Choice(['Fibonacci', 'Lucas', 'Pascal']))
@click.option('--n_iters', default=1, help='Number of iterations.')
@click.option('--grainsize', default=2, help='Size of grain used by sequences.')
def sequences(sequence, n_iters, grainsize):
    """Program that calculates a recursive sequence with the parallelism provided by Charm4Py."""

    print('Executing a ', sequence, 'sequence with', n_iters, 'iterations and grain size', grainsize, '.\n')

    charm.thisProxy.updateGlobals({'GRAINSIZE': grainsize}, awaitable=True).get()
    result = -1

    t0 = time.time()

    if sequence == 'Fibonacci':
        Group(FibUtil).compile(awaitable=True).get()  # precompile
        result = fibonacci(n_iters)

    if sequence == 'Lucas':
        Group(LucasUtil).compile(awaitable=True).get()  # precompile
        result = lucas(n_iters)

    print('')
    print('Result: ', result)
    print('Elapsed time: ', round(time.time() - t0, 3), 's', sep='')


def execute(args):
    print('')
    global GRAINSIZE
    sequences()

charm.start(execute)
