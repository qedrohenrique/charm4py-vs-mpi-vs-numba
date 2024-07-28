import matplotlib.pyplot as plt
import numpy as np

num_threads = [1, 2, 4, 8, 16]
charm4py_efficiency = [1, 0.6483, 0.3724, 0.1305, 0.0243]
mpi4py_efficiency = [1, 0.8748, 0.6197, 0.2810, 0.0490]
numba_efficiency = [1, 1.0875, 1.2157, 0.6237, 0.4052]

colors = ['#a3c68c', '#879676', '#6e6662', '#340735']

fig, ax = plt.subplots()

ax.plot(num_threads, charm4py_efficiency, marker='o', color=colors[0], label='Charm4Py', linestyle='-')
ax.plot(num_threads, mpi4py_efficiency, marker='o', color=colors[1], label='Mpi4Py', linestyle='-')
ax.plot(num_threads, numba_efficiency, marker='o', color=colors[2], label='Numba', linestyle='-')

ax.set_xlabel('Número de Threads/Processos')
ax.set_ylabel('Eficiência')
ax.legend()

ax.set_xticks(num_threads)
ax.set_xticklabels(num_threads)

ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()
