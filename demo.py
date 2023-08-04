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
    ti.init(arch=ti.gpu,  random_seed=int(time.time()))

    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT

    N = 500

    gui = ti.GUI("flocking behavior", res=(WINDOW_WIDTH, WINDOW_HEIGHT))

    rule = 0

    if rule == 0:
        alignment = gui.slider("align", 0, 20, 0.01)
        alignment.value = 1
        separation = gui.slider("separate", 0, 20, 0.01)
        separation.value = 1
        cohesion = gui.slider("cohere", 0, 20, 0.01)
        cohesion.value = 1
        rng = default_rng(seed=42)
        boid = Boid(N, 1e-2,
                    alignment.value, separation.value, cohesion.value,
                    1, 0.5,
                    distant=0.15, topo_num=7,
                    pos=rng.random(size=(N, 2), dtype=np.float32),
                    vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32)
                    )
        while gui.running:
            boid.get_neighbors(1)
            boid.wrapped()
            boid.update()
            boid.edge()
            boid.render(gui, AR)
    elif rule == 1:
        rng = default_rng(seed=42)
        viscek = Viscek(N, 1e-2,
                        35, 9,
                        distant=0.15,
                        pos=rng.random(size=(N, 2), dtype=np.float32),
                        vel=np.array([random_vector(2) for _ in range(N)], dtype=np.float32)
                        )

        while gui.running:
            viscek.get_neighbors(0)
            viscek.wrapped()
            viscek.update()
            viscek.edge()
            viscek.render(gui, AR)
