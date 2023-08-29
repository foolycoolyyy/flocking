import taichi as ti
import numpy as np
import time
import random
import math
from numpy.random import default_rng
from boid import Boid
from viscek import Viscek


def random_vector(n):
    components = [np.random.normal() for _ in range(n)]
    r = np.sqrt(sum(x * x for x in components))
    v = np.array([x / r for x in components])
    return v


if __name__ == "__main__":
    ti.init(arch=ti.gpu,  random_seed=int(time.time()), default_fp=ti.f64)

    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT

    N = 2000

    gui = ti.GUI("flocking behavior", res=(WINDOW_WIDTH, WINDOW_HEIGHT))

    rule = 1
    search_mode = 1

    if rule == 0:
        rng = default_rng(seed=42)
        boid = Boid(N, 1e-2,
                    2.0, 2.0, 2.0,
                    1, 0.5,
                    distant=0.1, topo_num=20,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32),
                    angle=2.5
                    )
        while gui.running:
            boid.get_neighbors(search_mode)
            boid.wrapped()
            boid.update()
            boid.edge()
            boid.render(gui, AR)
    elif rule == 1:
        rng = default_rng(seed=42)
        viscek = Viscek(N, 1,
                        0.01, 0.002, 0.005, 0.008,  # r0, rb, re, ra
                        0.4, 1.0,
                        distant=0.10, topo_num=12,
                        pos=rng.random(size=(N, 2), dtype=np.float32),
                        vel=np.array([random_vector(2)*0.01 for _ in range(N)], dtype=np.float32),
                        angle=1.0
                        )

        while gui.running:
            viscek.get_neighbors(search_mode)
            viscek.wrapped()
            viscek.update()
            viscek.edge()
            viscek.render(gui, AR)
