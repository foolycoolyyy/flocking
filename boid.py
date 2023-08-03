import taichi as ti
from flock import Flock
import numpy as np


@ti.data_oriented
class Boid(Flock):
    def __init__(self, num, dt, ali, sep, coh, max_spd, max_acc, distant=None, topo_num=None, pos=None, vel=None, acc=None):
        super().__init__(num, dt, distant, topo_num, pos, vel, acc)
        self.ali = ali
        self.sep = sep
        self.coh = coh
        self.max_spd = max_spd
        self.max_acc = max_acc
        print("args:", "num", self.num, "dt", self.dt, "ali", self.ali, "sep", self.sep, "coh", self.coh, "max_spd", self.max_spd,
              "max_acc",self.max_acc)

    @ti.func
    def set_mag(self, v, mag):
        return (v / v.norm()) * mag

    @ti.func
    def limit(self, v, mag):
        norm = v.norm()
        return (v / norm) * mag if norm > 0 and norm > mag else v

    @ti.func
    def clear_acc(self):
        for i in range(self.num):
            self.acceleration[i] = ti.Vector([0.0 for _ in range(2)])

    @ti.kernel
    def compute_force(self):
        self.clear_acc()
        for i in range(self.num):
            alignment = ti.Vector([0.0 for _ in range(2)])
            separation = ti.Vector([0.0 for _ in range(2)])
            cohesion = ti.Vector([0.0 for _ in range(2)])
            n = self.neighbors_num[i]
            for index in range(n):
                j = self.neighbors[i, index]
                alignment += self.velocity[j]
                separation += (self.position[i] - self.position[j]) / self.distant
                cohesion += self.position[j]
            if n > 0:
                alignment = self.limit(
                    self.set_mag((alignment / n), self.max_spd) - self.velocity[i],
                    self.max_acc) * self.ali
                separation = self.limit(
                    self.set_mag((separation / n), self.max_spd) - self.velocity[i],
                    self.max_acc) * self.sep
                cohesion = self.limit(
                    self.set_mag(((cohesion / n) - self.position[i]), self.max_spd) -
                    self.velocity[i], self.max_acc) * self.coh
                self.acceleration[i] += alignment
                self.acceleration[i] += separation
                self.acceleration[i] += cohesion
