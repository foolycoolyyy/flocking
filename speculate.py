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
        self.flock.change_topo_num(nc_gauss)
        self.flock.change_distant(distant_gauss)
        self.flock.update_neighbor_finder()
        self.flock.get_neighbors(search_mode)
        self.constract_n()
        self.constract_A()
        self.copy_to_numpy(self.A_np, self.A)
        self.compute_nc()
        self.compute_C_int()
        self.compute_J()
        self.compute_entropy()

    @ti.kernel
    def compute_nc(self):
        val = 0.0
        for i in range(self.flock.num):
            val += self.flock.neighbors_num[i]
        self.nc[None] = val / self.flock.num
        print("finish compute nc")

    @ti.kernel
    def compute_J(self):
        self.J[None] = 1 / (self.nc[None] / 2 * (1 - self.C_int[None]))
        print("finish compute J")

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
        print("finish constract n")

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
        print("finish constract A")

    def compute_entropy(self):
        self.eig, v = np.linalg.eig(self.A_np)
        log_Z = self.flock.num * self.J[None] * self.nc[None] / 2
        for i in range(1, self.flock.num):
            log_Z -= np.log(self.J[None] * self.eig[i])
        self.entropy[None] = -log_Z + 0.5 * self.J[None] * self.flock.num * self.nc[None] * self.C_int[None]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.flock.num):
            for j in range(self.flock.num):
                np_arr[i, j] = src_arr[i, j]


if __name__ == "__main__":
    ti.init(arch=ti.gpu, random_seed=int(time.time()), default_fp=ti.f64)
    N = 512
    rule = 0
    advanced_num = 3000
    search_mode = 1
    rng = default_rng(seed=42)
    viscek = Viscek(N, 1e-2,
                    0.01, 0.002, 0.005, 0.008,  # r0, rb, re, ra
                    0.4, 1.0,
                    distant=0.15, topo_num=0,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32)
                    )
    for i in range(advanced_num):
        viscek.get_neighbors(search_mode)
        viscek.wrapped()
        viscek.update()
        viscek.edge()
    print("finish advanced")
    size = 90
    entropy = np.zeros(size)
    topo_num = np.arange(5, size + 5)
    for topo_num_guass in range(5, size + 5):
        speculater = Speculate(viscek, search_mode, viscek.distant, topo_num_guass)
        entropy[topo_num_guass-5] = speculater.entropy[None]
        print(speculater.entropy[None])
    plt.plot(topo_num, entropy)
    plt.show()

