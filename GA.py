from graph import Graph, Node, DistanceMatrix

from population import Population, Tour

import crossover as co
import selection as se

import random
import copy

import argparse

import time
import logging


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

            after_select = time.time()

            logging.info('Selection: {}'.format(after_select - start_time))

            child1, child2 = cls._crossover(parent1, parent2)

            after_crossover = time.time()
            logging.info('Crossover: {}'.format(
                after_crossover - after_select))

            cls._mutate(child1)
            cls._mutate(child2)

            after_mutation = time.time()
            logging.info('Mutation: {}'.format(
                after_mutation - after_crossover))

            new_population.append_tour(child1)
            new_population.append_tour(child2)

            offset += 2

        return new_population

    @classmethod
    def _crossover(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
        return co.crossover_CX2(parent1, parent2)

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
    def _select(cls, population: Population) -> (Tour, Tour):
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
    if verbose:
        print(list(map(lambda node: node.id, path)))

    print(length)

    # Save answer
    filename = file_path.split('/')[-1]
    filename = filename[:-4]

    with open('sol_{}.csv'.format(filename), 'w') as f:
        f.writelines(list(map(lambda node: str(node.id) + '\n', path)))
