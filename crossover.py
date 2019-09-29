import random

from population import Tour
from graph import Graph

import copy

def crossover_order(parent1: Tour, parent2: Tour) -> (Tour, Tour):
    child1 = Tour()
    child2 = Tour()
    N = Graph.num_nodes

    start_index = random.randint(0, N)
    end_index = random.randint(0, N)

    if start_index > end_index:
        temp = start_index
        start_index = end_index
        end_index = temp

    node_ids1 = []
    node_ids2 = []

    for i in range(start_index, end_index):
        node1 = parent1.get_node(i)
        node_ids1.append(node1.id)

        node2 = parent2.get_node(i)
        node_ids2.append(node2.id)

        child1.add_node(i, node1)
        child2.add_node(i, node2)

    def binary_search(target, data):
        data.sort()
        start = 0
        end = len(data) - 1

        while start <= end:
            mid = (start + end) // 2

            if data[mid] == target:
                return True
            elif data[mid] < target:
                start = mid + 1
            else:
                end = mid - 1

        return False

    remaining_indices = list(range(end_index, N)) + \
        list(range(0, start_index))

    offset1 = 0
    offset2 = 0

    indices = list(range(0, N))
    cyclic_indices = indices[end_index:N] + indices[0:end_index]

    for i in cyclic_indices:
        node1 = parent2.get_node(i)
        node2 = parent1.get_node(i)
        # if child1.contains_node(node):
        if not binary_search(node1.id, node_ids1):
            child1.add_node(remaining_indices[offset1], node1)
            offset1 += 1

        if not binary_search(node2.id, node_ids2):
            child2.add_node(remaining_indices[offset2], node2)
            offset2 += 1

    child1.update_distance()
    child2.update_distance()

    return child1, child2

def crossover_CX2(parent1: Tour, parent2: Tour) -> (Tour, Tour):
    # TODO: Need Refactoring

    child1 = Tour()
    child2 = Tour()

    visited_p1 = [False] * Graph.num_nodes
    visited_p2 = [False] * Graph.num_nodes
    child_index = 0

    p1 = parent1
    p2 = parent2

    remaining_c1 = []
    remaining_c2 = []

    while True:
        if len(visited_p2) == 0:
            # Corner case: no cycle in a step
            for i in range(len(remaining_c1)):
                child1.add_node(-i-1, Graph.nodes_by_id[remaining_c1[i]])

            for i in range(len(remaining_c2)):
                child2.add_node(-i-1, Graph.nodes_by_id[remaining_c2[i]])

            return child1, child2

        # Step 2
        node = p2.get_node(0)
        visited_p2[0] = True
        child1.add_node(0 + child_index, node)

        # Step 3:
        pos = p1.id_to_position[node.id]
        visited_p1[pos] = True
        pos2 = p1.id_to_position[p2.get_node(pos).id]

        node2 = p2.get_node(pos2)
        visited_p2[pos2] = True
        child2.add_node(0 + child_index, node2)

        i = 1
        c1_nodes = [node.id]
        c2_nodes = [node2.id]

        while i + child_index < Graph.num_nodes:
            if node2.id == p1.get_node(0).id:
                visited_p1[0] = True
                if len(set(c1_nodes) - set(c2_nodes)) != 0:
                    for id in c1_nodes:
                        if id in c2_nodes:
                            continue
                        remaining_c2.append(id)

                    for id in c2_nodes:
                        if id in c1_nodes:
                            continue
                        remaining_c1.append(id)

                break

            # Step 4:
            pos = p1.id_to_position[node2.id]
            visited_p1[pos] = True
            node = p2.get_node(pos)

            visited_p2[pos] = True

            child1.add_node(i + child_index, node)
            c1_nodes.append(node.id)

            # Repeat Step 3:
            pos = p1.id_to_position[node.id]
            visited_p1[pos] = True

            pos2 = p1.id_to_position[p2.get_node(pos).id]
            node2 = p2.get_node(pos2)

            visited_p2[pos2] = True

            child2.add_node(i + child_index, node2)
            c2_nodes.append(node2.id)

            i += 1

            if node2.id == p1.get_node(0).id:
                visited_p1[0] = True
                if len(set(c1_nodes) - set(c2_nodes)) != 0:
                    for id in c1_nodes:
                        if id in c2_nodes:
                            continue
                        remaining_c2.append(id)
                    for id in c2_nodes:
                        if id in c1_nodes:
                            continue
                        remaining_c1.append(id)

                break

        child_index += i
        if child_index == Graph.num_nodes:
            return child1, child2

        new_p1 = Tour()
        new_p2 = Tour()

        i1 = 0
        i2 = 0
        for pos in range(len(visited_p1)):
            if not visited_p1[pos]:
                new_p1.add_node(i1, p1.get_node(pos))
                i1 += 1
            if not visited_p2[pos]:
                new_p2.add_node(i2, p2.get_node(pos))
                i2 += 1

        assert i1 == i2

        visited_p1 = [False] * i1
        visited_p2 = [False] * i2

        p1 = new_p1
        p2 = new_p2

def crossover_edge_recombination(parent1: Tour, parent2: Tour) -> (Tour, Tour):
    # Adjacency Information
    # Node 1 -> (-9, 2, 4)
    # Node 2 -> (1, -3, 5)
    # (-) for both adjacent element
    adjacency = {}
    for i in range(Graph.num_nodes):
        node = parent1.get_node(i)
        adjacency[node.id] = [parent1.get_node(i-1).id,
                                parent1.get_node(i+1).id]

    for i in range(Graph.num_nodes):
        node = parent2.get_node(i)
        ids = [parent2.get_node(i-1).id, parent2.get_node(i+1).id]

        neighbors = adjacency[node.id]

        for id in ids:
            if id in neighbors:
                i = neighbors.index(id)
                adjacency[node.id][i] = -1 * neighbors[i]
            else:
                adjacency[node.id].append(id)

    adjacencies = [adjacency, copy.deepcopy(adjacency)]

    parents = [parent1, parent2]
    children = [Tour(), Tour()]

    def select_neighbor_id(prev_id, adjacency):
        ids = adjacency[prev_id]
        selected_id = []
        for id in ids:
            if id < 0:
                selected_id = [-id]
                break

        if len(selected_id) == 0:
            min_num_neighbors = 10**10
            for id in ids:
                if min_num_neighbors > len(adjacency[id]):
                    selected_id = [id]
                    min_num_neighbors = len(adjacency[id])
                if min_num_neighbors == len(adjacency[id]):
                    selected_id.append(id)

        if len(selected_id) == 0:
            min_num_neighbors = 10**10
            for id in adjacency.keys():
                if id == prev_id:
                    continue
                if min_num_neighbors > len(adjacency[id]):
                    selected_id = [id]
                    min_num_neighbors = len(adjacency[id])
                if min_num_neighbors == len(adjacency[id]):
                    selected_id.append(id)

        assert len(selected_id) != 0

        if len(selected_id) > 1:
            return selected_id[random.randint(0, len(selected_id)-1)]
        return selected_id[0]

    def delete_node_from_all_neighbors(prev_id, adjacency):
        for id in adjacency.keys():
            try:
                adjacency[id].remove(prev_id)
            except ValueError:
                pass
            try:
                adjacency[id].remove(-1 * prev_id)
            except ValueError:
                pass

    def delete_node_from_table(prev_id, adjacency):
        del adjacency[prev_id]

    for i in range(2):
        child = children[i]
        parent = parents[i]

        adjacency = adjacencies[i]

        child.add_node(0, parent.get_node(0))
        prev_id = child.get_node(0).id

        for i in range(1, Graph.num_nodes):
            delete_node_from_all_neighbors(prev_id, adjacency)

            selected_id = select_neighbor_id(prev_id, adjacency)
            delete_node_from_table(prev_id, adjacency)
            child.add_node(i, Graph.get_node(selected_id))

            prev_id = selected_id

    return children[0], children[1]

def crossover_inverse_sequence(cls, parent1: Tour, parent2: Tour) -> (Tour, Tour):
    pass
