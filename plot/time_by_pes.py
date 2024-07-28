import matplotlib.pyplot as plt
import numpy as np

num_threads = [1, 2, 4, 8, 16]
serial_times = [2.0355, 2.0355, 2.0355, 2.0355, 2.0355]
charm4py_times = [2.4318, 1.8755, 1.6326, 2.3292, 6.2458]
mpi4py_times = [2.4744, 1.4143, 0.9982, 1.1009, 3.1557]
numba_times = [2.5545, 0.9963, 0.5253, 0.512, 0.394]

bar_width = 0.2
index = np.arange(len(num_threads))
colors = ['#a3c68c', '#879676', '#6e6662', '#340735']

fig, ax = plt.subplots()
bar1 = ax.bar(index, serial_times, bar_width, label='Serial', color=colors[0])
bar2 = ax.bar(index + bar_width, charm4py_times, bar_width, label='Charm4Py', color=colors[1])
bar3 = ax.bar(index + 2 * bar_width, mpi4py_times, bar_width, label='Mpi4Py', color=colors[2])
bar4 = ax.bar(index + 3 * bar_width, numba_times, bar_width, label='Numba', color=colors[3])

ax.set_xlabel('Número de Threads/Processos')
ax.set_ylabel('Tempo de Execução')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(num_threads)
ax.legend()

ax.grid(True, which='both', linestyle='--', linewidth=0.2)

plt.show()
