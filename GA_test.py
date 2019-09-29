import pytest

import copy

import crossover as co
import selection as se

from graph import Graph, Node, DistanceMatrix
from GA import GA
from population import Population, Tour


@pytest.fixture()
def graph():
    # setup
    nodes = [Node(i+1) for i in range(5)]
    distanceMatrix = DistanceMatrix(5, init_matrix=[
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    Graph.set_graph(nodes, distanceMatrix)

    yield "resource"
    # teardown
    pass


def test_population(graph):
    population = Population(5)

    for tour in population:
        # print(list(map(lambda node: node.id, tour.path)))
        # print(tour.distance)
        assert len(tour.path) == len(Graph.nodes)


def test_crossover(graph):
    parent1 = Tour(is_random=True)
    parent2 = Tour(is_random=True)

    child1, child2 = GA._crossover(parent1, parent2)

    assert sorted(child1.path) == sorted(Graph.nodes)
    assert sorted(child2.path) == sorted(Graph.nodes)


def test_rank_based_selection(graph):
    population = Population(10)

    a = population.get_rank_probability(2)

    assert len(a) == 10
    assert abs(sum(a) - 1) < 0.001

    parents = se.select_roulette_sampling(population, num_samples=2, s=1.5)

    assert len(parents) == 2


def test_edge_recombination_crossover(graph):
    parent1 = Tour(is_random=True)
    parent2 = Tour(is_random=True)

    child1, child2 = co.crossover_edge_recombination(parent1, parent2)

    assert sorted(child1.path) == sorted(Graph.nodes)
    assert sorted(child2.path) == sorted(Graph.nodes)


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
    child1, child2 = co.crossover_CX2(parent1, parent2)

    assert sorted(child1.path) == sorted(nodes)
    assert sorted(child2.path) == sorted(nodes)


def test_mutate(graph):
    tour = Tour(is_random=True)

    old_tour = copy.deepcopy(tour)

    GA._mutate(tour)

    assert sorted(old_tour.path) == sorted(tour.path)
