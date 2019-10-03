from graph import Graph, Node, DistanceMatrix

from population import Population, Tour

from greedy import solve_greedy

import crossover as co
import selection as se
import mutation as mu

from typing import List

import random
import copy

import argparse

import time
import logging


class GA:
    mutation_rate = 0.015
    adaptable_mutation_rate = False
    elitism = True
    tournament_size = 5
    selection_pressure = 1.5    # 1 to 2

    @classmethod
    def evolve_population(cls, population: Population) -> Population:
        if cls.adaptable_mutation_rate:
            if cls.mutation_rate > 0.001:
                cls.mutation_rate -= 0.0005

            logging.info('mutation rate: {}'.format(cls.mutation_rate))

        new_population = Population()
        N = len(population)
        offset = 0
        if cls.elitism:
            new_population.append_tour(population.get_fittest())
            offset += 1

        while offset < N:
            start_time = time.time()

            parents = cls._select(population)

            after_select = time.time()

            child1, child2 = cls._crossover(parents[0], parents[1])

            after_crossover = time.time()

            cls._mutate(child1)
            cls._mutate(child2)

            after_mutation = time.time()

            new_population.append_tour(child1)
            new_population.append_tour(child2)

            offset += 2

        return new_population

    @classmethod
    def _crossover(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        return co.crossover_order(parent1, parent2)

    @classmethod
    def _mutate(cls, tour: Tour):
        mu.mutate_swap_connections(
            tour, mutation_rate=cls.mutation_rate, only_better=False)

    @classmethod
    def _select(cls, population: Population) -> List[Tour]:
        # return se.select_roulette_sampling(population, s=cls.selection_pressure)
        return se.select_tournament(population, cls.tournament_size)


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
    import inspect

    problem = TSP()
    mode = "geo" if args.geo else "euc2d"
    problem.from_file(file_path, mode=mode)

    logging.info('file: {}'.format(file_path))
    logging.info('population size: {}'.format(population_size))
    logging.info('fitness_evaluations: {}'.format(fitness_evaluations))
    logging.info(inspect.getsource(GA._crossover))
    logging.info(inspect.getsource(GA._select))
    logging.info(inspect.getsource(GA._mutate))

    def solve_GA(problem, generations):
        Graph.set_graph(problem.nodes, problem.distance_matrix)

        tours = []
        seed = Tour(path=solve_greedy(problem))
        tours.append(seed)
        for _ in range(population_size - 1):
            tour = copy.deepcopy(seed)
            mu.mutate_swap_connections(tour, mutation_rate=1)

            tours.append(tour)

        population = Population(population_size, tours=tours)

        for i in range(generations):
            logging.info('\n{}th generation'.format(i))

            if verbose:
                print('{}th generation'.format(i))

            population = GA.evolve_population(population)

            best_tour = population.get_fittest()
            logging.info('distance: {}'.format(best_tour.distance))

            if verbose:
                print('distance: {}\n'.format(best_tour.distance))

        best_tour = population.get_fittest()

        return best_tour.path, best_tour.distance

    path, length = solve_GA(problem, fitness_evaluations)

    assert sorted(path) == sorted(Graph.nodes)

    # Print ans
    if verbose:
        print(list(map(lambda node: node.id, path)))

    print(length)

    # Save answer
    filename = file_path.split('/')[-1]
    filename = filename[:-4]

    with open('sol_{}.csv'.format(filename), 'w') as f:
        f.writelines(list(map(lambda node: str(node.id) + '\n', path)))
