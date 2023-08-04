import taichi as ti
import sys
import numpy as np
from numpy.random import default_rng
from flock import Flock
import math


@ti.func
def random_vector():
    angle = ti.random(ti.f32) * 2 * math.pi
    v = ti.Vector([ti.cos(angle), ti.sin(angle)])
    return v


@ti.func
def normalized(v):
    return v / v.norm()


@ti.data_oriented
class Viscek(Flock):
    def __init__(self, num, dt, alpha, beta, distant=None, topo_num=None, pos=None, vel=None, acc=None):
        super().__init__(num, dt, distant, topo_num, pos, vel, acc)
        self.v0 = ti.field(ti.f32, shape=())
        self.alpha = alpha
        self.beta = beta
        self.forward_vel = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)

    @ti.kernel
    def check(self):
        norm = self.velocity[0].norm()
        self.v0[None] = norm

    @ti.kernel
    def clear_acc(self):
        for i in range(self.num):
            self.acceleration[i] = ti.Vector([0.0 for _ in range(2)])

    @ti.kernel
    def compute_vel(self):
        for i in range(self.num):
            sum_vel = ti.Vector([0.0 for _ in range(2)])
            n = self.neighbors_num[i]
            for index in range(n):
                j = self.neighbors[i, index]
                sum_vel += self.velocity[j]
            self.forward_vel[i] = self.v0[None] * normalized(self.alpha * sum_vel + n * random_vector())

        for i in range(self.num):
            self.velocity[i] = self.forward_vel[i]

    def wrapped(self):
        self.check()
        self.clear_acc()
        self.compute_vel()

