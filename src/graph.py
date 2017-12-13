import edge


class Graph(dict):
    """
        An exemplary graph structure:
        {"A": {"B": Edge("A", "B", 1), "C": Edge("A", "C", 2)},
        "B": {"C": Edge("B", "C", 3), "D": Edge("B", "D", 4)},
        "C": {"D": Edge("C", "D", 5)},
        "D": {"C": Edge("D", "C", 6)},
        "E": {"C": Edge("E", "C", 7)},
        "F": {}}
        """
    def __init__(self):
        pass

    def v(self):
        """Return the number of nodes (the graph order)."""
        num=len(self)
        return num

    def e(self):
        """Return the number of edges in O(V) time."""
        edges = sum(len(self[node]) for node in self)
        return edges

    def add_node(self, node):
        """Add a node to the graph."""
        if node not in self:
            self[node] = dict()

    def add_edge(self, edge):
        """Add an edge to the graph (missing nodes are created)."""
        self.add_node(edge.source)
        self.add_node(edge.target)
        self[edge.source][edge.target] = edge

    def iternodes(self):
        """Generate all nodes from the graph on demand."""
        return self.iterkeys()

    def iteredges(self):
        """Generate all edges from the graph on demand."""
        for source in self.iternodes():
            for target in self[source]:
                yield self[source][target]

    def iteradjacent(self, source):
        """Generate the adjacent nodes from the graph on demand."""
        return self[source].iterkeys()

    def iteroutedges(self, source):
        """Generate the outedges from the graph on demand."""
        for target in self[source]:
            yield self[source][target]

    def iterinedges(self, source):
        """Generate the inedges from the graph on demand."""
        for target in self.iternodes():
            if source in self[target]:
                yield self[target][source]



    def show(self):
        """The graph presentation."""
        for source in self.iternodes():
            print(source, ":")
            for edge in self.iteroutedges(source):
                print("%s(%s)" % (edge.target, edge.weight))
            print()




