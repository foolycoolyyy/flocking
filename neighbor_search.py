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

    @ti.func
    def partition_with_indices(self, arr, indices, left, right):
        pivot_value = arr[indices[right]]
        i = left - 1

        for j in range(left, right):
            if arr[indices[j]] <= pivot_value:
                i += 1
                indices[i], indices[j] = indices[j], indices[i]

        indices[i + 1], indices[right] = indices[right], indices[i + 1]
        return i + 1, indices

    @ti.func
    def k_smallest_with_indices(self, arr):
        k = self.topo_num
        left, right = 0, len(arr) - 1
        indices = ti.Vector([i for i in range(len(arr))])
        while left <= right:
            pivot_index, indices = self.partition_with_indices(arr, indices, left, right)

            if pivot_index == k - 1:
                break
            elif pivot_index > k - 1:
                right = pivot_index - 1
            else:
                left = pivot_index + 1
        ret = ti.Vector([0 for _ in range(100)])
        for i in range(k):
            ret[i] = indices[i]
        return ret

    @ti.kernel
    def topo_search(self):
        for i in range(self.num):
            cnt = 0
            distant = ti.Vector([0.0 for _ in range(self.num - 1)])
            index = ti.Vector([0 for _ in range(self.num - 1)])
            for j in range(self.num):
                if j != i:
                    distant[cnt] = (self.position[i] - self.position[j]).norm()
                    index[cnt] = j
                    cnt += 1
            result_indices = self.k_smallest_with_indices(distant)
            for j in range(self.topo_num):
                self.neighbors_num[i] = self.topo_num
                self.neighbors[i, result_indices[j]] = index[j]
