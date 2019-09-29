import pytest

import copy

from graph import Node, DistanceMatrix
from GA import Population, GA, Graph, Tour


def test_population():
    nodes = [Node(i+1) for i in range(5)]
    distanceMatrix = DistanceMatrix(5, init_matrix=[
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    Graph.set_graph(nodes, distanceMatrix)
    population = Population(5)

    for tour in population:
        # print(list(map(lambda node: node.id, tour.path)))
        # print(tour.distance)
        assert len(tour.path) == len(nodes)


def test_crossover():
    nodes = [Node(i+1) for i in range(5)]
    distanceMatrix = DistanceMatrix(5, init_matrix=[
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    Graph.set_graph(nodes, distanceMatrix)
    parent1 = Tour(is_random=True)
    parent2 = Tour(is_random=True)
    # print(parent1.path)
    # print(parent2.path)

    child1, child2 = GA._crossover(parent1, parent2)

    assert sorted(child1.path) == sorted(nodes)
    assert sorted(child2.path) == sorted(nodes)
    # print(path_to_id(child.path))

def test_edge_recombination_crossover():
    nodes = [Node(i+1) for i in range(5)]
    distanceMatrix = DistanceMatrix(5, init_matrix=[
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    Graph.set_graph(nodes, distanceMatrix)
    parent1 = Tour(is_random=True)
    parent2 = Tour(is_random=True)
    # print(parent1.path)
    # print(parent2.path)

    child1, child2 = GA._crossover_edge_recombination(parent1, parent2)

    assert sorted(child1.path) == sorted(nodes)
    assert sorted(child2.path) == sorted(nodes)
    print(path_to_id(child1.path))
    print(path_to_id(child2.path))


def test_CX2_crossover():
    nodes = [Node(i+1) for i in range(8)]
    distanceMatrix = DistanceMatrix(8)

    Graph.set_graph(nodes, distanceMatrix)
    # parent1 = Tour(path=[3,4,8,2,7,1,6,5])
    # parent2 = Tour(path=[4,2,5,1,6,8,3,7])
    # parent1 = Tour(path=[1,2,3,4,5,6,7,8])
    # parent2 = Tour(path=[2,7,5,8,4,1,6,3])
    parent1 = Tour(path=[7, 6, 3, 8, 5, 1, 4, 2])
    parent2 = Tour(path=[3, 8, 2, 4, 5, 1, 7, 6])
    child1, child2 = GA._crossover_CX2(parent1, parent2)

    assert sorted(child1.path) == sorted(nodes)
    assert sorted(child2.path) == sorted(nodes)
    print(path_to_id(child1.path))
    print(path_to_id(child2.path))

def test_mutate():
    nodes = [Node(i+1) for i in range(5)]
    distanceMatrix = DistanceMatrix(5, init_matrix=[
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    Graph.set_graph(nodes, distanceMatrix)
    tour = Tour(is_random=True)

    old_tour = copy.deepcopy(tour)

    GA._mutate(tour)

    # print(old_tour.path)
    # print(tour.path)

    assert sorted(old_tour.path) == sorted(tour.path)


def path_to_id(path):
    return list(map(lambda node: node.id, path))
