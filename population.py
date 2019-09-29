from graph import Graph, Node, DistanceMatrix

from typing import List

import random

import copy

# Individual = Tour
class Tour:
    def __init__(self, is_random=False, path=None):
        assert Graph.num_nodes > 0

        self.distance = 0
        self.id_to_position = {}

        if path and len(path) == Graph.num_nodes:
            path = list(map(lambda id: Node(id=id), path))

        if is_random:
            path = copy.deepcopy(Graph.nodes)
            random.shuffle(path)

        if path == None:
            self.path = [None] * Graph.num_nodes
            return

        for i in range(Graph.num_nodes):
            self.id_to_position[path[i].id] = i

        prev_node = path[0]
        for node in path[1:] + path[:1]:
            self.distance += Graph.distanceMatrix.getDistance(
                node, prev_node)
            prev_node = node

        self.path = path

    def __len__(self):
        return Graph.num_nodes

    def __repr__(self):
        return str(self._path_to_id(self.path))

    def add_node(self, index, node):
        self.path[index] = node
        self.id_to_position[node.id] = index

    def get_node(self, index):
        if index >= Graph.num_nodes:
            index -= Graph.num_nodes
        return self.path[index]

    def contains_node(self, n2):
        for n1 in self.path:
            if n1 == None:
                continue

            if n1.id == n2.id:
                return True

        return False

    def update_distance(self):
        self.distance = 0

        prev_node = self.path[0]
        for node in self.path[1:] + self.path[:1]:
            self.distance += Graph.distanceMatrix.getDistance(
                node, prev_node)
            prev_node = node

    def _path_to_id(self, path):
        return list(map(lambda node: node.id if node != None else -1, path))


class Population:
    def __init__(self, population_size: int = 0):
        self.tours = []

        for _ in range(population_size):
            # generate initial individuals
            random_tour = Tour(is_random=True)
            self.tours.append(random_tour)

    def __len__(self):
        return len(self.tours)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        if self.index + 1 >= len(self.tours):
            raise StopIteration
        else:
            self.index += 1
            return self.tours[self.index]

    def append_tour(self, tour: Tour):
        self.tours.append(tour)

    def remove_tour(self, tour: Tour):
        self.tours.remove(tour)

    def get_tour(self, index: int):
        if index < 0 or index >= len(self.tours):
            return

        return self.tours[index]

    def get_fittest(self) -> Tour:
        min_distance = float('inf')
        best_tour = None

        for tour in self.tours:
            if tour.distance < min_distance:
                min_distance = tour.distance
                best_tour = tour

        assert best_tour != None

        return best_tour

    def get_rank_probability(self, s) -> List[Tour]:
        assert 1 <= s <= 2
        # rank based selection probability

        sorted_tour = sorted(enumerate(self.tours), key=lambda t: t[1].distance)

        mu = len(self)

        P = [0] * mu

        for i in range(mu):
            index = sorted_tour[i][0]
            P[index]= ((2 - s) / mu) \
                + (i * (s - 1) / sum(range(mu)))
        
        return P
