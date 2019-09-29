import random
from population import Population, Tour

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

    return selected[0], selected[1]

def select_roulette_sampling(population: Population):
    pass
