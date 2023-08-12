import taichi as ti
import numpy as np
import time
import random
import math
from numpy.random import default_rng
from boid import Boid
from viscek import Viscek
from flock import Flock
from demo import random_vector
import matplotlib.pyplot as plt


def normalized(v):
    return v / v.norm()


@ti.func
def normalized_ti(v):
    return v / v.norm()


@ti.data_oriented
class Speculate:
    def __init__(self, flock, search_mode, distant_gauss=None, nc_gauss=None):
        self.flock = flock
        self.nc_gauss = nc_gauss
        self.distant_gauss = distant_gauss
        self.nc = ti.field(dtype=ti.f64, shape=())
        # in topo search the neighbors number else the average neighbors number
        self.C_int = ti.field(dtype=ti.f64, shape=())
        self.J = ti.field(dtype=ti.f64, shape=())
        self.A = ti.field(dtype=ti.f64, shape=(self.flock.num, self.flock.num))
        self.A_np = np.ndarray((self.flock.num, self.flock.num), dtype=np.float64)
        self.eig = np.zeros(self.flock.num)
        self.n = ti.field(dtype=ti.f64, shape=(self.flock.num, self.flock.num))
        self.n.fill(0.0)
        self.entropy = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def compute_nc(self):
        val = 0.0
        for i in range(self.flock.num):
            val += self.flock.neighbors_num[i]
        self.nc[None] = val / self.flock.num

    @ti.kernel
    def compute_J(self):
        self.J[None] = 1 / (self.nc[None] / 2 * (1 - self.C_int[None]))

    @ti.kernel
    def compute_C_int(self):
        ret = 0.0
        for i in range(self.flock.num):
            n = self.flock.neighbors_num[i]
            for index in range(n):
                j = self.flock.neighbors[i, index]
                ret += ti.math.dot(normalized_ti(self.flock.velocity[i]),
                                   normalized_ti(self.flock.velocity[j])
                                   )
        self.C_int[None] = ret / self.flock.num / self.nc[None]

    @ti.kernel
    def constract_n(self):
        for i in range(self.flock.num):
            for index in range(self.flock.neighbors_num[i]):
                j = self.flock.neighbors[i, index]
                self.n[i, j] += 0.5
                self.n[j, i] += 0.5

    @ti.kernel
    def constract_A(self):
        for i in range(self.flock.num):
            for j in range(self.flock.num):
                if i == j:
                    temp = 0.0
                    for k in range(self.flock.num):
                        temp += self.n[i, k]
                    self.A[i, j] = temp - self.n[i, j]
                else:
                    self.A[i, j] = -self.n[i, j]

    def compute_entropy(self):
        self.eig, v = np.linalg.eig(self.A_np)
        log_Z = self.flock.num * self.J[None] * self.nc[None] / 2
        for i in range(1, self.flock.num):
            if self.eig[i] > 0:
                log_Z -= np.log(self.J[None] * self.eig[i])
        self.entropy[None] = -log_Z + 0.5 * self.J[None] * self.flock.num * self.nc[None] * self.C_int[None]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.flock.num):
            for j in range(self.flock.num):
                np_arr[i, j] = src_arr[i, j]

    def change_nc_gauss(self, n):
        self.nc_gauss = n

    def change_distant_gauss(self, dis):
        self.distant_gauss = dis

    def wrapped(self):
        self.flock.change_topo_num(self.nc_gauss)
        self.flock.change_distant(self.distant_gauss)
        self.flock.update_neighbor_finder()
        self.flock.get_neighbors(search_mode)
        self.constract_n()
        self.constract_A()
        self.copy_to_numpy(self.A_np, self.A)
        self.compute_nc()
        self.compute_C_int()
        self.compute_J()
        self.compute_entropy()


if __name__ == "__main__":
    ti.init(arch=ti.gpu, random_seed=int(time.time()), default_fp=ti.f64, device_memory_GB=2)
    N = 512
    advanced_num = 3000
    search_mode = 1
    total_size = 15
    total_begin = 3
    sim_per_arg = 5
    begin = 5
    nc_sim = np.arange(total_begin, total_begin + total_size)
    nc_mem = np.zeros(total_size, dtype=int)
    err = np.zeros(total_size)
    rng = default_rng(seed=42)
    viscek = Viscek(N, 1e-2,
                    0.01, 0.002, 0.005, 0.008,  # r0, rb, re, ra
                    0.3, 1.0,
                    distant=0.15, topo_num=nc_sim[0],
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32),
                    angle=2.0
                    )
    speculater = Speculate(viscek, search_mode, viscek.distant, begin)
    size = 50
    entropy = np.zeros(size)
    topo_num = np.arange(begin, size + begin)
    max_entropy = 0.0
    max_entropy_topo_num = 0
    nc_per_arg = np.zeros(sim_per_arg)
    for k in range(total_size):
        viscek.change_topo_num(nc_sim[k])
        for j in range(sim_per_arg):
            for i in range(advanced_num):
                viscek.get_neighbors(search_mode)
                viscek.wrapped()
                viscek.update()
                viscek.edge()
            for topo_num_gauss in range(begin, size + begin):
                speculater.change_nc_gauss(topo_num_gauss)
                speculater.wrapped()
                entropy[topo_num_gauss-begin] = speculater.entropy[None]
            for i in range(size):
                if entropy[i] > max_entropy:
                    max_entropy = entropy[i]
                    max_entropy_topo_num = i + begin
            nc_per_arg[j] = max_entropy_topo_num
        err[k] = np.std(nc_per_arg) / np.sqrt(len(nc_per_arg))
        nc_mem[k] = np.sum(nc_per_arg) / len(nc_per_arg)

    plt.errorbar(nc_sim, nc_mem, err, ecolor='k', elinewidth=0.5, marker='o', mfc='blue',
                 mec='k', mew=1, ms=10, alpha=1, capsize=5, capthick=3, linestyle="none")
    plt.xlabel("J_sim")
    plt.ylabel("J_mem")
    plt.show()
