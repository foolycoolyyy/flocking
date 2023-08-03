import taichi as ti


@ti.data_oriented
class NeighborSearch:
    def __init__(self,
                 neighbor_num_max,
                 num,
                 position,
                 distant=None,
                 topo_num=None,
                 ):
        self.neighbor_num_max = neighbor_num_max
        self.num = num
        self.position = position
        self.distant = distant
        self.topo_num = topo_num
        self.neighbors = ti.field(int, shape=(num, neighbor_num_max))
        self.neighbors.fill(-1)
        self.neighbors_num = ti.field(int, shape=num)

    @ti.kernel
    def distant_search(self):
        for i in range(self.num):
            cnt = 0
            for j in range(self.num):
                if (self.position[i] - self.position[j]).norm() < self.distant and i != j:
                    self.neighbors[i, cnt] = j
                    cnt += 1
            self.neighbors_num[i] = cnt
