from numba import jit
import numpy as np
import time

MAX_ITER = 10_000
THRESHOLD = 0.0001
INITIAL_ERR = 1_000_000.0
MATRIX_SIZE = 512+2


@jit()
def compute_stencil(a, a_new, m, n):
    current_error = 0.0
    for p in range(2, (m - 2)):
        for q in range(2, (n - 2)):
            a_new[p][q] = 0.25 * (a[p][q + 1] + a[p][q - 1] + a[p - 1][q] + a[p + 1][q])
            current_error = max(current_error, abs(a_new[p][q] - a[p][q]))

    return current_error


a = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
a_new = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
err = INITIAL_ERR
iters = 0

for i in range(0, MATRIX_SIZE):
    a[1][i] = 1.0
    a_new[1][i] = 1.0

for i in range(0, MATRIX_SIZE):
    a[MATRIX_SIZE - 2][i] = 1.0
    a_new[MATRIX_SIZE - 2][i] = 1.0

for i in range(0, MATRIX_SIZE):
    a[i][1] = 1.0
    a_new[i][1] = 1.0

for i in range(0, MATRIX_SIZE):
    a[i][MATRIX_SIZE - 2] = 1.0
    a_new[i][MATRIX_SIZE - 2] = 1.0

start_time = time.time()
while err > THRESHOLD and iters < MAX_ITER:
    err = compute_stencil(a, a_new, MATRIX_SIZE, MATRIX_SIZE)
    a_new, a = a, a_new
    iters += 1

end_time = (time.time() - start_time)

print("\n[%dx%d]" % (MATRIX_SIZE - 2, MATRIX_SIZE - 2))
print("%d iterations" % iters)
print("Final error: ", err)
print("Elapsed time: %.4f seconds" % end_time)
print("%.4f\n" % end_time)
