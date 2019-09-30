from graph import Position, Node, DistanceMatrix

import logging


class Edge():
    def __init__(self, n1, n2, distance):
        self.n1 = n1
        self.n2 = n2
        self.distance = distance

    def __repr__(self):
        return '({}, {}) => {}'.format(self.n1.id, self.n2.id, self.distance)


def optimize(tsp, edges):
    pass

# TODO
# Slightly better O(N^2) algorithm for converting travelling edge set to path


def edges_to_path(edges):
    connections = []
    pass


def solve_greedy(tsp):
    print('FINISH: Read file')

    N = len(tsp.nodes)
    nodes = tsp.nodes
    distanceMatrix = tsp.distanceMatrix

    if N < 1:
        return [], 0

    edges = [Edge(nodes[i], nodes[j], distanceMatrix.get_distance(
        nodes[i], nodes[j]))
        for i in range(N) for j in range(i)]

    print('FINISH: Collect all edges')

    edges.sort(key=lambda x: x.distance)

    print('FINISH: Sort edges by distance')

    # connect edges in order if possible

    connected_edges = []
    append_edge = connected_edges.append

    connected_sets = [[nodes[i]] for i in range(N)]
    count_sets = N

    # from node id to connected set index
    set_indices = {id: id-1 for id in range(1, N+1)}

    for e in edges:
        n1 = e.n1
        n2 = e.n2

        if n1.connected == 2 or n2.connected == 2:
            continue

        n1_connected_set = set_indices[n1.id]
        n2_connected_set = set_indices[n2.id]

        if count_sets > 1 and n1_connected_set == n2_connected_set:
            continue

        e.n1.connected += 1
        e.n2.connected += 1

        append_edge(e)

        if count_sets == 1:
            break

        def merge_sets(set1, set2):
            try:
                # for python >= 3.5
                return [*set1, *set2]
            except SyntaxError:
                return set1 + set2

        # merge n2_connected_set => n1_connected_set
        connected_sets[n1_connected_set] \
            = merge_sets(connected_sets[n1_connected_set],
                         connected_sets[n2_connected_set])

        for node_id in set_indices.keys():
            if set_indices[node_id] == n2_connected_set:
                set_indices[node_id] = n1_connected_set

        count_sets -= 1

    print('FINISH: connect possible short edges')
    print('distance_edge_sum: {}'.format(
        sum(map(lambda e: e.distance, connected_edges))))

    adj_matrix = [[0]*(N+1) for _ in range(N+1)]

    for e in connected_edges:
        adj_matrix[e.n1.id][e.n2.id] = e.distance
        adj_matrix[e.n2.id][e.n1.id] = e.distance

    print('FINISH: make adjacency matrix')

    distance = 0

    # O(N^2)
    path = [tsp.nodes[0]]
    remaining_node = N-1
    while remaining_node > 0:
        n_id = path[-1].id
        for i in range(1, N+1):
            if adj_matrix[n_id][i]:
                distance += adj_matrix[n_id][i]

                adj_matrix[i][n_id] = 0
                path.append(tsp.nodes[i-1])

                remaining_node -= 1
                break

    distance += adj_matrix[path[0].id][path[-1].id]

    print('distance_path: {}'.format(distance))
    return path


if __name__ == '__main__':
    from tsp_solver import TSP
    tsp = TSP()
    tsp.from_file('problems/rl11849.tsp')
    path = solve_greedy(tsp)

    print(path)

    # Save answer
    filename = 'rl11849'

    with open('sol_{}_greedy.csv'.format(filename), 'w') as f:
        f.writelines(list(map(lambda node: str(node.id) + '\n', path)))
