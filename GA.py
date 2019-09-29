from graph import Node, DistanceMatrix

from typing import List

import random
import copy

import argparse

import time
import logging


class Graph:
    @classmethod
    def set_graph(cls, nodes, distanceMatrix):
        cls.nodes = nodes
        cls.num_nodes = len(nodes)
        cls.distanceMatrix = distanceMatrix

        cls.nodes_by_id = {}
        for node in nodes:
            cls.nodes_by_id[node.id] = node

    @classmethod
    def get_node(cls, id):
        return cls.nodes_by_id[id]


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
        start_time = time.time()

        prev_node = self.path[0]
        for node in self.path[1:] + self.path[:1]:
            self.distance += Graph.distanceMatrix.getDistance(
                node, prev_node)
            prev_node = node

        logging.info('Updating distances: {}'.format(time.time() - start_time))

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


class GA:
    mutation_rate = 0.03
    elitism = True
    tournament_size = 5

    @classmethod
    def evolve_population(cls, population: Population) -> Population:
        # if cls.mutation_rate > 0.001:
        #     cls.mutation_rate -= 0.0005

        print('mutation rate: {}'.format(cls.mutation_rate))

        new_population = Population()
        N = len(population)
        offset = 0
        if cls.elitism:
            new_population.append_tour(population.get_fittest())
            offset += 1

        while offset < N:
            logging.info('Create One Child')
            start_time = time.time()

            parent1, parent2 = cls._select(population)
            # print(parent1)
            # print(parent2)
            # print('\n')

            after_select = time.time()

            logging.info('Selection: {}'.format(after_select - start_time))

            child1, child2 = cls._crossover(parent1, parent2)

            after_crossover = time.time()
            logging.info('Crossover: {}'.format(
                after_crossover - after_select))

            cls._mutate(child1)
            cls._mutate(child2)

            # print(child1.distance)
            # print(child2.distance)

            after_mutation = time.time()
            logging.info('Mutation: {}'.format(
                after_mutation - after_crossover))

            new_population.append_tour(child1)
            new_population.append_tour(child2)

            offset += 2

        return new_population

    @classmethod
    def _crossover(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        return cls._crossover_CX2(parent1, parent2)

    @classmethod
    def _crossover_order(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        child1 = Tour()
        child2 = Tour()
        N = Graph.num_nodes

        start_index = random.randint(0, N)
        end_index = random.randint(0, N)

        if start_index > end_index:
            temp = start_index
            start_index = end_index
            end_index = temp

        node_ids1 = []
        node_ids2 = []

        for i in range(start_index, end_index):
            node1 = parent1.get_node(i)
            node_ids1.append(node1.id)

            node2 = parent2.get_node(i)
            node_ids2.append(node2.id)

            child1.add_node(i, node1)
            child2.add_node(i, node2)

        def binary_search(target, data):
            data.sort()
            start = 0
            end = len(data) - 1

            while start <= end:
                mid = (start + end) // 2

                if data[mid] == target:
                    return True
                elif data[mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1

            return False

        remaining_indices = list(range(end_index, N)) + \
            list(range(0, start_index))

        offset1 = 0
        offset2 = 0

        indices = list(range(0, N))
        cyclic_indices = indices[end_index:N] + indices[0:end_index]

        for i in cyclic_indices:
            node1 = parent2.get_node(i)
            node2 = parent1.get_node(i)
            # if child1.contains_node(node):
            if not binary_search(node1.id, node_ids1):
                child1.add_node(remaining_indices[offset1], node1)
                offset1 += 1

            if not binary_search(node2.id, node_ids2):
                child2.add_node(remaining_indices[offset2], node2)
                offset2 += 1

        child1.update_distance()
        child2.update_distance()

        return child1, child2

    @classmethod
    def _crossover_CX2(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        # TODO: Need Refactoring

        child1 = Tour()
        child2 = Tour()

        visited_p1 = [False] * Graph.num_nodes
        visited_p2 = [False] * Graph.num_nodes
        child_index = 0

        p1 = parent1
        p2 = parent2

        remaining_c1 = []
        remaining_c2 = []

        while True:
            if len(visited_p2) == 0:
                # Corner case: no cycle in a step
                for i in range(len(remaining_c1)):
                    child1.add_node(-i-1, Graph.nodes_by_id[remaining_c1[i]])

                for i in range(len(remaining_c2)):
                    child2.add_node(-i-1, Graph.nodes_by_id[remaining_c2[i]])

                return child1, child2

            # Step 2
            node = p2.get_node(0)
            visited_p2[0] = True
            child1.add_node(0 + child_index, node)

            # Step 3:
            pos = p1.id_to_position[node.id]
            visited_p1[pos] = True
            pos2 = p1.id_to_position[p2.get_node(pos).id]

            node2 = p2.get_node(pos2)
            visited_p2[pos2] = True
            child2.add_node(0 + child_index, node2)

            i = 1
            c1_nodes = [node.id]
            c2_nodes = [node2.id]

            while i + child_index < Graph.num_nodes:
                if node2.id == p1.get_node(0).id:
                    visited_p1[0] = True
                    if len(set(c1_nodes) - set(c2_nodes)) != 0:
                        for id in c1_nodes:
                            if id in c2_nodes:
                                continue
                            remaining_c2.append(id)

                        for id in c2_nodes:
                            if id in c1_nodes:
                                continue
                            remaining_c1.append(id)

                    break

                # Step 4:
                pos = p1.id_to_position[node2.id]
                visited_p1[pos] = True
                node = p2.get_node(pos)

                visited_p2[pos] = True

                child1.add_node(i + child_index, node)
                c1_nodes.append(node.id)

                # Repeat Step 3:
                pos = p1.id_to_position[node.id]
                visited_p1[pos] = True

                pos2 = p1.id_to_position[p2.get_node(pos).id]
                node2 = p2.get_node(pos2)

                visited_p2[pos2] = True

                child2.add_node(i + child_index, node2)
                c2_nodes.append(node2.id)

                i += 1

                if node2.id == p1.get_node(0).id:
                    visited_p1[0] = True
                    if len(set(c1_nodes) - set(c2_nodes)) != 0:
                        for id in c1_nodes:
                            if id in c2_nodes:
                                continue
                            remaining_c2.append(id)
                        for id in c2_nodes:
                            if id in c1_nodes:
                                continue
                            remaining_c1.append(id)

                    break

            child_index += i
            if child_index == Graph.num_nodes:
                return child1, child2

            new_p1 = Tour()
            new_p2 = Tour()

            i1 = 0
            i2 = 0
            for pos in range(len(visited_p1)):
                if not visited_p1[pos]:
                    new_p1.add_node(i1, p1.get_node(pos))
                    i1 += 1
                if not visited_p2[pos]:
                    new_p2.add_node(i2, p2.get_node(pos))
                    i2 += 1

            assert i1 == i2

            visited_p1 = [False] * i1
            visited_p2 = [False] * i2

            p1 = new_p1
            p2 = new_p2

    @classmethod
    def _crossover_edge_recombination(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        # Adjacency Information
        # Node 1 -> (-9, 2, 4)
        # Node 2 -> (1, -3, 5)
        # (-) for both adjacent element
        adjacency = {}
        for i in range(Graph.num_nodes):
            node = parent1.get_node(i)
            adjacency[node.id] = [parent1.get_node(i-1).id,
                                  parent1.get_node(i+1).id]

        for i in range(Graph.num_nodes):
            node = parent2.get_node(i)
            ids = [parent2.get_node(i-1).id, parent2.get_node(i+1).id]

            neighbors = adjacency[node.id]

            for id in ids:
                if id in neighbors:
                    i = neighbors.index(id)
                    adjacency[node.id][i] = -1 * neighbors[i]
                else:
                    adjacency[node.id].append(id)

        adjacencies = [adjacency, copy.deepcopy(adjacency)]

        parents = [parent1, parent2]
        children = [Tour(), Tour()]

        def select_neighbor_id(prev_id, adjacency):
            ids = adjacency[prev_id]
            selected_id = []
            for id in ids:
                if id < 0:
                    selected_id = [-id]
                    break

            if len(selected_id) == 0:
                min_num_neighbors = 10**10
                for id in ids:
                    if min_num_neighbors > len(adjacency[id]):
                        selected_id = [id]
                        min_num_neighbors = len(adjacency[id])
                    if min_num_neighbors == len(adjacency[id]):
                        selected_id.append(id)

            if len(selected_id) == 0:
                min_num_neighbors = 10**10
                for id in adjacency.keys():
                    if id == prev_id:
                        continue
                    if min_num_neighbors > len(adjacency[id]):
                        selected_id = [id]
                        min_num_neighbors = len(adjacency[id])
                    if min_num_neighbors == len(adjacency[id]):
                        selected_id.append(id)

            assert len(selected_id) != 0

            if len(selected_id) > 1:
                return selected_id[random.randint(0, len(selected_id)-1)]
            return selected_id[0]

        def delete_node_from_all_neighbors(prev_id, adjacency):
            for id in adjacency.keys():
                try:
                    adjacency[id].remove(prev_id)
                except ValueError:
                    pass
                try:
                    adjacency[id].remove(-1 * prev_id)
                except ValueError:
                    pass

        def delete_node_from_table(prev_id, adjacency):
            del adjacency[prev_id]

        for i in range(2):
            child = children[i]
            parent = parents[i]

            adjacency = adjacencies[i]

            child.add_node(0, parent.get_node(0))
            prev_id = child.get_node(0).id

            for i in range(1, Graph.num_nodes):
                delete_node_from_all_neighbors(prev_id, adjacency)

                selected_id = select_neighbor_id(prev_id, adjacency)
                delete_node_from_table(prev_id, adjacency)
                child.add_node(i, Graph.get_node(selected_id))

                prev_id = selected_id

        return children[0], children[1]

    @classmethod
    def _crossover_inverse_sequence(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        pass

    @classmethod
    def _mutate(cls, tour: Tour):
        N = Graph.num_nodes
        for i in range(N):
            if random.random() < cls.mutation_rate:
                swap_index = random.randint(0, N-1)

                temp = tour.get_node(i)
                tour.add_node(i, tour.get_node(swap_index))
                tour.add_node(swap_index, temp)

        tour.update_distance()

    @classmethod
    # Use Tornament Selection
    def _select(cls, population: Population) -> (Tour, Tour):
        selected = []
        while len(selected) < 2:
            tournament = Population()
            for _ in range(cls.tournament_size):
                random_index = random.randint(0, len(population)-1)

                tournament.append_tour(population.get_tour(random_index))

            selected_tour = tournament.get_fittest()

            # def is_equal(t1, t2):
            #     for i in range(Graph.num_nodes):
            #         if t1.path[i] != t2.path[i]:
            #             return False
            #     return True

            # if len(selected) == 1 and is_equal(selected[0], selected_tour):
            #     continue

            selected.append(selected_tour)

        return selected[0], selected[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filepath', help="set file path containing TSP")
    parser.add_argument("-p", "--population",
                        help="set initial population size",
                        type=int, default=50)
    parser.add_argument("-f", "--evaluations",
                        help="set number of fitness evaluations",
                        type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    parser.add_argument("--geo", action="store_true",
                        help="use geo distance")

    args = parser.parse_args()

    verbose = args.verbose
    file_path = args.filepath
    population_size = args.population
    fitness_evaluations = args.evaluations

    logging.basicConfig(filename='run{}.log'.format(int(time.time())),
                        filemode='w', level=logging.INFO)

    from tsp_solver import TSP

    problem = TSP()
    mode = "geo" if args.geo else "euc2d"
    problem.from_file(file_path, mode=mode)

    def solve_GA(problem, generations):
        Graph.set_graph(problem.nodes, problem.distanceMatrix)
        population = Population(population_size)

        for i in range(generations):
            if verbose:
                print('{}th generation'.format(i))

            population = GA.evolve_population(population)

            if verbose:
                best_tour = population.get_fittest()
                print('distance: {}\n'.format(best_tour.distance))

        best_tour = population.get_fittest()

        return best_tour.path, best_tour.distance

    path, length = solve_GA(problem, fitness_evaluations)

    assert sorted(path) == sorted(Graph.nodes)

    # Print ans
    print(list(map(lambda node: node.id, path)))
    print(length)

    # Save answer
    filename = file_path.split('/')[-1]
    filename = filename[:-4]

    with open('sol_{}.csv'.format(filename), 'w') as f:
        f.writelines(list(map(lambda node: str(node.id) + '\n', path)))
