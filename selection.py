import random
from population import Population, Tour
from typing import List


def select_tournament(population: Population, tournament_size) -> (Tour, Tour):
    selected = []
    while len(selected) < 2:
        tournament = Population()
        for _ in range(tournament_size):
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

    return selected


def select_roulette_sampling(population: Population, num_samples=2, s: int = 1.5) -> List[Tour]:
    p = population.get_rank_probability(s)

    N = len(population)

    # a: cumulative probability
    a = [p[0]] + [0] * (N-1)
    for i in range(1, N):
        a[i] = a[i-1] + p[i]

    r = random.uniform(0, 1/num_samples)

    i = 0
    mating_pool = []

    while len(mating_pool) < num_samples:
        while a[i] < r:
            i += 1

        mating_pool.append(population.get_tour(i))
        r += 1 / num_samples

    return mating_pool
