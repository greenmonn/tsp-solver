from itertools import permutations

import time

from graph import Position, Node, DistanceMatrix

MAX_NUMBER = 10 ** 10


# TODO: TSP by GNN


class TSP():
    def __init__(self):
        self.distanceMatrix = None
        self.nodes = []
        self.timestamp = str(time.time())

    def from_array(self, distance_array):
        N = len(distance_array)

        self.nodes = [Node(id) for id in range(1, N+1)]

        self.distanceMatrix = DistanceMatrix(N, distance_array)

    def from_file(self, file_name, mode="euc2d"):
        with open(file_name) as f:
            while True:
                line = f.readline()
                line = line.strip()

                if not line:
                    break

                if not line[0].isdigit():
                    # not a node
                    continue

                node = list(filter(
                    lambda x: x != '',
                    line.split(' ')))

                id = int(node[0])
                x, y = map(float, node[1:])

                self.nodes.append(Node(id, x, y))

        # generate distance matrix N * N
        N = len(self.nodes)
        self.distanceMatrix = DistanceMatrix(N)

        for i in range(N):
            for j in range(i):
                n1 = self.nodes[i]
                n2 = self.nodes[j]

                self.distanceMatrix.setDistance(n1, n2, mode)


def solve_exhaustive(tsp):
    # find all permutation of nodes (N!)
    # choose path with minimum length
    if len(tsp.nodes) < 1:
        return []

    start_node = tsp.nodes[0]
    all_paths = permutations(tsp.nodes[1:])

    min_path_length = MAX_NUMBER
    min_path = None
    for path in all_paths:
        path = (start_node,) + path + (start_node,)
        path_length = _get_path_length(path, tsp.distanceMatrix)

        if path_length < min_path_length:
            min_path_length = path_length
            min_path = path

    return min_path[:-1], min_path_length


def solve_dp(tsp):
    if len(tsp.nodes) < 1:
        return [], 0

    # Optimization 1: Fix starting node
    start_index = 0
    start_node = tsp.nodes[start_index]

    N = len(tsp.nodes)

    # Optimization 2: Memoize M(i)(visited)
    path, length = _get_shortest_path(
        start=start_node, last=start_node, visited=(1 << start_index),
        tsp=tsp)

    path = [start_node] + path

    return path, length


def memoize(func):
    cache = {}

    def memoizer(*args, **kwargs):
        key = str(kwargs['last'].id) + \
            str(kwargs['visited']) + str(kwargs['tsp'])

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return memoizer


@memoize
def _get_shortest_path(start, last, visited, tsp):
    D = tsp.distanceMatrix
    nodes = tsp.nodes
    N = len(D)

    if visited == ((1 << N) - 1):
        return [], D.getDistance(last, start)

    min_length = MAX_NUMBER
    path = None

    for i in range(len(D)):
        if visited & (1 << i) != 0:
            continue

        distance = D.getDistance(last, nodes[i])
        subpath, length = _get_shortest_path(start=start, last=nodes[i],
                                             visited=visited | (1 << i), tsp=tsp)
        length = length + distance

        if length < min_length:
            min_length = length
            path = [nodes[i]] + subpath

    return path, min_length


def _get_path_length(path, distanceMatrix):
    if len(path) < 1:
        return 0

    prev_node = path[0]
    length = 0

    for node in path[1:]:
        length += distanceMatrix.getDistance(node, prev_node)
        prev_node = node

    return length
