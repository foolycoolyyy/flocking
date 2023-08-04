import taichi as ti
from flock import Flock


@ti.func
def normalized(v):
    return v / v.norm()


