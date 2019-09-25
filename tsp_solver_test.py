import pytest

from tsp_solver import TSP, solve_exhaustive, solve_dp


def test_read_file():
    problem = TSP()
    problem.from_file('problems/bier127.tsp')

    assert len(problem.nodes) == 127
    assert len(problem.distanceMatrix.matrix) == 127


def test_tsp_solve():
    #  Prepare the square symmetric distance matrix for 3 nodes:
    #  Distance from A to B is 1.0
    #                B to C is 3.0
    #                A to C is 2.0
    problem = TSP()

    problem.from_array([
        [],
        [1.0],
        [3.0, 2.0],
        [5.0, 4.0, 1.0]
    ])

    path, length = solve_exhaustive(problem)

    assert list(map(lambda node: node.id, path)) == [1, 2, 3, 4]
    assert length == 9.0


def test_city_5():
    problem = TSP()

    problem.from_array([
        [],
        [3.0],
        [4.0, 4.0],
        [2.0, 6.0, 5.0],
        [7.0, 3.0, 8.0, 6.0]
    ])

    path, length = solve_dp(problem)

    assert list(map(lambda node: node.id, path)) == [1, 3, 2, 5, 4]
    assert length == 19.0


def test_burma_14():
    problem = TSP()

    problem.from_file('problems/burma14.tsp', 'geo')

    assert len(problem.nodes) == 14

    path, length = solve_dp(problem)

    assert list(map(lambda node: node.id, path)) == [
        1, 2, 14, 3, 4, 5, 6, 12, 7, 13, 8, 11, 9, 10]
    assert int(length) == 3346


def test_city_15():
    problem = TSP()

    problem.from_array([[],
                        [29],
                        [82, 55],
                        [46, 46, 68],
                        [68, 42, 46, 82],
                        [52, 43, 55, 15, 74],
                        [72, 43, 23, 72, 23, 61],
                        [42, 23, 43, 31, 52, 23, 42],
                        [51, 23, 41, 62, 21, 55, 23, 33],
                        [55, 31, 29, 42, 46, 31, 31, 15, 29],
                        [29, 41, 79, 21, 82, 33, 77, 37, 62, 51],
                        [74, 51, 21, 51, 58, 37, 37, 33, 46, 21, 65],
                        [23, 11, 64, 51, 46, 51, 51, 33, 29, 41, 42, 61],
                        [72, 52, 31, 43, 65, 29, 46, 31, 51, 23, 59, 11, 62],
                        [46, 21, 51, 64, 23, 59, 33, 37, 11, 37, 61, 55, 23, 59]])

    path, length = solve_dp(problem)

    ans = [13, 2, 15, 9, 5, 7, 3, 12, 14, 10, 8, 6, 4, 11, 1]
    ans.reverse()
    assert list(map(lambda node: node.id, path)) == ans
    assert length == 291
