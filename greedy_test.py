import pytest

from graph import Graph, Node, DistanceMatrix
from greedy import solve_greedy

from tsp_solver import TSP


@pytest.fixture()
def tsp():
    # setup
    tsp = TSP()
    tsp.from_array([
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    return tsp


def test_greedy(tsp):
    path = solve_greedy(tsp)

    print(list(map(lambda node: node.id, path)))
    assert sorted(path) == sorted(tsp.nodes)


def test_greedy_large(tsp):
    tsp = TSP()
    tsp.from_file('problems/fl1400.tsp')
    path = solve_greedy(tsp)

    assert sorted(path) == sorted(tsp.nodes)
