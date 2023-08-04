import taichi as ti
import numpy as np
from neighbor_search import NeighborSearch


@ti.data_oriented
class Flock:
    def __init__(self, num, dt, distant=None, topo_num=None, pos=None, vel=None, acc=None):
        neighbor_num_max = 1000
        self.num = num
        self.dt = dt
        self.distant = distant
        self.topo_num = topo_num
        self.position = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)
        self.velocity = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)
        self.acceleration = ti.Vector.field(n=2, dtype=ti.f32, shape=self.num)
        self.neighbors = ti.field(int, shape=(num, neighbor_num_max))
        self.neighbors_num = ti.field(int, shape=num)
        self.neighbor_finder = NeighborSearch(neighbor_num_max, self.num, self.position, self.distant, self.topo_num)

        self.init_field(self.position, pos)
        self.init_field(self.velocity, vel)
        self.init_field(self.acceleration, acc)

    def get_neighbors(self, search_mode):
        if search_mode == 0:
            self.neighbor_finder.distant_search()
        elif search_mode == 1:
            self.neighbor_finder.topo_search()
        self.neighbors = self.neighbor_finder.neighbors
        self.neighbors_num = self.neighbor_finder.neighbors_num

    def init_field(self, field, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                field.from_numpy(value)
            else:
                field.from_numpy(
                    np.full(fill_value=value, dtype=np.float32, shape=self.n))

    @ti.kernel
    def edge(self):
        for i in range(self.num):
            if self.position[i].x < 0:
                self.position[i].x += 1.
            if self.position[i].x > 1:
                self.position[i].x -= 1.
            if self.position[i].y < 0:
                self.position[i].y += 1.
            if self.position[i].y > 1:
                self.position[i].y -= 1.

    @ti.kernel
    def update(self):
        for i in range(self.num):
            self.velocity[i] += self.dt * self.acceleration[i]
            self.position[i] += self.dt * self.velocity[i]

    def render(self, gui, size=0.02, filename=None):
        gui.clear(0xffffff)
        centers = self.position.to_numpy()
        gui.circles(centers, color=0x000000, radius=1)
        if filename is None:
            gui.show()
        else:
            gui.show(filename)
