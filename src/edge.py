
class Edge:

    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight

    def __repr__(self):
        """Compute the string representation of the edge."""
        return "(%s,%s)" % (
                repr(self.source),
                repr(self.target))

    def __repr__(self):
        """Compute the string representation of the edge."""
        return "(%s,%s,%s)" % (
                repr(self.source),
                repr(self.target),
                repr(self.weight))