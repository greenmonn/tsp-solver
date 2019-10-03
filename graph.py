from geopy.distance import geodesic
from math import sqrt


class Graph:
    @classmethod
    def set_graph(cls, nodes, distance_matrix):
        cls.nodes = nodes
        cls.num_nodes = len(nodes)
        cls.distance_matrix = distance_matrix

        cls.nodes_by_id = {}
        for node in nodes:
            cls.nodes_by_id[node.id] = node

    @classmethod
    def get_node(cls, id):
        return cls.nodes_by_id[id]


class Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, position):
        return sqrt((self.x - position.x)**2 + (self.y - position.y)**2)

    def distance_geo(self, position):
        return geodesic((self.x, self.y), (position.x, position.y)).km

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)


class DistanceMatrix():
    def __init__(self, N, init_matrix=None):
        if init_matrix != None:
            self.matrix = init_matrix
        else:
            self.matrix = [[0] * i for i in range(N)]

    def set_distance(self, n1, n2, mode):
        if n1.id <= n2.id:
            return

        if mode == 'euc2d':
            self.matrix[n1.id - 1][n2.id - 1] = n1.distance(n2)
        elif mode == 'geo':
            self.matrix[n1.id - 1][n2.id - 1] = n1.distance_geo(n2)

    def get_distance(self, n1, n2):
        if n1.id == n2.id:
            return 0

        if n1.id < n2.id:
            tmp = n2
            n2 = n1
            n1 = tmp

        return self.matrix[n1.id - 1][n2.id - 1]

    def __len__(self):
        return len(self.matrix)


class Node():
    def __init__(self, id, x=None, y=None):
        self.id = id
        if x == None or y == None:
            self.position = None
        else:
            self.position = Position(x, y)
        self.connected = []
        self.connected_num = 0

    def __eq__(self, other):
        if other is None:
            return False

        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def distance(self, node):
        if self.position == None:
            return None

        return self.position.distance(node.position)

    def distance_geo(self, node):
        if self.position == None:
            return None

        return self.position.distance_geo(node.position)

    def __repr__(self):
        return 'Node({}: {})'.format(self.id, self.position)
