from numba import jit
import numpy as np
import time

MAX_M = 512 + 2
MAX_ITER = 100000
MAX_TOL = 0.0001
INITIAL_ERR = 1000000.0


@jit()
def compute_stencil(a, a_new, m, n):
    current_error = 0.0
    for p in range(2, (m - 2)):
        for q in range(2, (n - 2)):
            a_new[p][q] = 0.25 * (a[p][q + 1] + a[p][q - 1] + a[p - 1][q] + a[p + 1][q])
            current_error = max(current_error, abs(a_new[p][q] - a[p][q]))

    return current_error


a = np.zeros((MAX_M, MAX_M))
a_new = np.zeros((MAX_M, MAX_M))
err = INITIAL_ERR
iters = 0

for i in range(0, MAX_M):
    a[1][i] = 1.0
    a_new[1][i] = 1.0

for i in range(0, MAX_M):
    a[MAX_M - 2][i] = 1.0
    a_new[MAX_M - 2][i] = 1.0

for i in range(0, MAX_M):
    a[i][1] = 1.0
    a_new[i][1] = 1.0

for i in range(0, MAX_M):
    a[i][MAX_M - 2] = 1.0
    a_new[i][MAX_M - 2] = 1.0

start_time = time.time()
while err > MAX_TOL and iters < MAX_ITER:
    if iters % 2 == 0:
        err = compute_stencil(a, a_new, MAX_M, MAX_M)
    else:
        err = compute_stencil(a_new, a, MAX_M, MAX_M)
    iters += 1

end_time = (time.time() - start_time)

print("\n[%dx%d]" % (MAX_M - 2, MAX_M - 2))
print("%d iterations" % iters)
print("Final error: ", err)
print("Elapsed time: %.4f seconds" % end_time)
print("%.4f\n" % end_time)
