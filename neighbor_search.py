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
        self.distant0 = distant
        self.topo_num = topo_num
        self.neighbors = ti.field(int, shape=(num, neighbor_num_max))
        self.neighbors.fill(-1)
        self.neighbors_num = ti.field(int, shape=num)

        # topo_search temp
        self.distant = ti.field(ti.f32, shape=(num, num-1))
        self.index = ti.field(int, shape=(num, num-1))
        self.indices = ti.field(int, shape=(num, num-1))

    @ti.kernel
    def distant_search(self):
        for i in range(self.num):
            cnt = 0
            for j in range(self.num):
                if (self.position[i] - self.position[j]).norm() < self.distant0 and i != j:
                    self.neighbors[i, cnt] = j
                    cnt += 1
            self.neighbors_num[i] = cnt

    @ti.func
    def partition_with_indices(self, m, left, right):
        pivot_value = self.distant[m, self.indices[m, right]]
        i = left - 1

        for j in range(left, right):
            if self.distant[m, self.indices[m, j]] <= pivot_value:
                i += 1
                t = self.indices[m, i]
                self.indices[m, i] = self.indices[m, j]
                self.indices[m, j] = t
        t = self.indices[m, i+1]
        self.indices[m, i+1] = self.indices[m, right]
        self.indices[m, right] = t
        return i + 1

    @ti.func
    def init_indices(self, i):
        for j in range(self.num - 1):
            self.indices[i, j] = j

    @ti.func
    def k_smallest_with_indices(self, i):
        left, right = 0, self.num - 1 - 1
        self.init_indices(i)
        while left <= right:
            pivot_index = self.partition_with_indices(i, left, right)

            if pivot_index == self.topo_num - 1:
                break
            elif pivot_index > self.topo_num - 1:
                right = pivot_index - 1
            else:
                left = pivot_index + 1

    @ti.kernel
    def topo_search(self):
        for i in range(self.num):
            cnt = 0
            for j in range(self.num):
                if j != i:
                    self.distant[i, cnt] = (self.position[i] - self.position[j]).norm()
                    self.index[i, cnt] = j
                    cnt += 1
            self.k_smallest_with_indices(i)
            for j in range(self.topo_num):
                self.neighbors_num[i] = self.topo_num
                self.neighbors[i, j] = self.index[i, self.indices[i, j]]

