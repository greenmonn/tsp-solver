from graph import Position, Node, DistanceMatrix

import logging


class Edge():
    def __init__(self, n1, n2, distance):
        self.n1 = n1
        self.n2 = n2
        self.distance = distance

    def __repr__(self):
        return '({}, {}) => {}'.format(self.n1.id, self.n2.id, self.distance)


def optimize(nodes, path, distance_matrix):
    N = len(nodes)
    D = distance_matrix.get_distance

    def get_node_from_path(path, index, N):
        while index >= N:
            index -= N
        return path[index]

    def set_node_in_path(path, index, N, node):
        if index >= N:
            index -= N

        path[index] = node

    for i in range(N):
        for gap in range(3, N-1):
            # swap connections [a-b, c-d] => [a-c, b-d] if better
            a = get_node_from_path(path, i, N)
            b = get_node_from_path(path, i+1, N)
            c = get_node_from_path(path, i+gap, N)
            d = get_node_from_path(path, i+gap+1, N)

            if D(a, b) + D(c, d) > D(a, c) + D(b, d):
                nodes[a.id-1].connected.remove(b)
                nodes[a.id-1].connected.append(c)
                nodes[b.id-1].connected.remove(a)
                nodes[b.id-1].connected.append(d)

                nodes[c.id-1].connected.remove(d)
                nodes[c.id-1].connected.append(a)
                nodes[d.id-1].connected.remove(c)
                nodes[d.id-1].connected.append(b)

                path[:], distance = make_path(nodes, distance_matrix)

    return distance


def calculate_distance(path, matrix):
    prev = path[0]
    distance = 0
    D = matrix.get_distance

    for node in path[1:] + path[:0]:
        distance += D(prev, node)
        prev = node

    return distance


# TODO
# Slightly better O(N^2) algorithm for converting travelling edge set to path


def make_path(nodes, distances):
    if len(nodes) == 0:
        return []

    prev_node = None
    node = nodes[0]
    path = [node]

    distance = 0

    while len(path) < len(nodes):
        for next in node.connected:
            if prev_node != None and next.id == prev_node.id:
                continue

            distance += distances.get_distance(node, next)

            prev_node = node
            node = next
            path.append(next)

            break

    distance += distances.get_distance(path[-1], path[0])

    return path, distance


def solve_greedy(tsp, optimize_count=3):
    print('FINISH: Read file')

    N = len(tsp.nodes)
    nodes = tsp.nodes
    distance_matrix = tsp.distance_matrix

    if N < 1:
        return [], 0

    edges = [Edge(nodes[i], nodes[j], distance_matrix.get_distance(
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

        if n1.degree == 2 or n2.degree == 2:
            continue

        n1_connected_set = set_indices[n1.id]
        n2_connected_set = set_indices[n2.id]

        if count_sets > 1 and n1_connected_set == n2_connected_set:
            continue

        e.n1.degree += 1
        e.n1.connected.append(e.n2)
        e.n2.degree += 1
        e.n2.connected.append(e.n1)

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

    path, distance = make_path(nodes, distance_matrix)

    print('distance_path: {}'.format(distance))

    for _ in range(optimize_count):
        distance = optimize(nodes, path, distance_matrix)

    print('optimized_distance: {}'.format(distance))

    for node in nodes:
        # reset nodes
        # TODO: use connected list just as local variable (only used in greedy module)
        node.connected = []
        node.degree = 0

    return path


if __name__ == '__main__':
    from tsp_solver import TSP
    tsp = TSP()
    filename = 'rl11849'

    tsp.from_file('problems/{}.tsp'.format(filename))
    path = solve_greedy(tsp)

    # print(list(map(lambda node: node.id, path)))

    # Save answer
    with open('sol_{}_greedy.csv'.format(filename), 'w') as f:
        f.writelines(list(map(lambda node: str(node.id) + '\n', path)))
