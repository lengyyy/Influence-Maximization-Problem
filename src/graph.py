

class Graph(dict):
    """
        An exemplary graph structure:
        {"A": {"B":1, "C": 2},
        "B": {"C": 3, "D": 4)},
        "C": {"D": 5},
        "D": {}}
        """
    def __init__(self):
        pass

    def v(self):
        """Return the number of nodes (the graph order)."""
        num = len(self)
        return num

    def e(self):
        """Return the number of edges in O(V) time."""
        edges = sum(len(self[node]) for node in self)
        return edges

    def add_node(self, node):
        """Add a node to the graph."""
        if node not in self:
            self[node] = dict()

    def add_edge(self, source, target, weight):
        """Add an edge to the graph (missing nodes are created)."""
        self.add_node(source)
        self.add_node(target)
        self[source][target] = weight

    def outdegree(self, source):
        """Return the outdegree of the node."""
        return len(self[source])

    def neighbor(self,source):
        return self[source].items()

    def neighbor_node(self,source):
        return self[source].keys()

    def del_node(self, source):
        """Remove a node from the graph (with edges)."""
        for target in self.keys():
            if source in self[target]:
                del self[target][source]
        del self[source]






