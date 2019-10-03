import random
from GA import Graph


def mutate_random_swap(tour, mutation_rate):
    N = Graph.num_nodes
    for i in range(N):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, N-1)

            temp = tour.get_node(i)
            tour.add_node(i, tour.get_node(swap_index))
            tour.add_node(swap_index, temp)

    tour.update_distance()


def mutate_swap_connections_gap_1(tour, mutation_rate):
    N = Graph.num_nodes
    D = Graph.distance_matrix.get_distance

    for i in range(N):
        if random.random() >= mutation_rate:
            continue

        # [a b _ c d] => [a c _ b d] if better
        a = tour.get_node(i)
        b = tour.get_node(i+1)
        c = tour.get_node(i+3)
        d = tour.get_node(i+4)

        if D(a, b) + D(c, d) > D(a, c) + D(b, d):
            tour.add_node(i+1, c)
            tour.add_node(i+3, b)

    tour.update_distance()


def mutate_swap_connections(tour, mutation_rate=1, only_better=True):
    N = Graph.num_nodes
    D = Graph.distance_matrix.get_distance

    if random.random() >= mutation_rate:
        return

    i = random.randint(0, N-1)
    gap = random.randint(1, N-4)

    # [a b _ c d] => [a c _ b d] if better
    a = tour.get_node(i)
    b = tour.get_node(i+1)
    c = tour.get_node(i+1+gap+1)
    d = tour.get_node(i+1+gap+2)

    if only_better and D(a, b) + D(c, d) <= D(a, c) + D(b, d):
        return

    s = i+1 + (i+1+gap+1)
    for index in range(i+1, i+1+((gap+2)//2)+1):
        swap_index = s - index
        if swap_index == index:
            continue

        temp = tour.get_node(index)
        tour.add_node(index, tour.get_node(swap_index))
        tour.add_node(swap_index, temp)

    tour.update_distance()
