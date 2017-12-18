import random
import time
import getopt
import sys
from multiprocessing import Process, Queue


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
            if source in self[target].keys():
                del self[target][source]
        del self[source]


def read_file(datafile, seedfile):
    """
    Read the network data file and seed data file, to get the graph and seed set.
    :param datafile: the absolute path of network file
    :param seedfile: the absolute path of seed set file
    """
    global n_nodes, n_edges , graph, seedlist
    lines = open(datafile).readlines()
    n_nodes = lines[0].split()[0]
    n_edges = lines[0].split()[1]
    for i in lines[1:]:
        thisline = i.split()
        graph.add_edge(int(thisline[0]), int(thisline[1]), float(thisline[2]))

    lines2 = open(seedfile).readlines()
    for i in lines2:
        seedlist.append(int(i))


def ise (q, times, model, random_seed):
    '''
    Influence spread estimation
    :param times: the run times
    :param model: The diffusion model: IC or LT
    :return: the average influence spread
    '''
    random.seed(random_seed)
    tem = []
    for i in range(times):
        tem.append(model())
    q.put(float(sum(tem))/len(tem))


def IC():
    '''
    Ise based on Independent Cascade model
    :return: the influence spread
    '''
    ActivitySet = seedlist[:]
    nodeActived = set(seedlist)
    count = len(ActivitySet)

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for neighbor, weight in graph.neighbor(seed):
                if neighbor not in nodeActived:
                    if random.random() < weight:
                        nodeActived.add(neighbor)
                        newActivitySet.append(neighbor)
        count=count+len(newActivitySet)
        ActivitySet = newActivitySet
    return count


def LT():
    '''
    ISE based on linear threshold model
    :return: the influence spread
    '''
    ActivitySet = seedlist[:]
    nodeActived = set(seedlist)
    count = len(ActivitySet)
    nodeThreshold = {}
    weights = {}

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for neighbor, weight in graph.neighbor(seed):
                if neighbor not in nodeActived:
                    if neighbor not in nodeThreshold:
                        nodeThreshold[neighbor] = random.random()
                        weights[neighbor] = 0
                    weights[neighbor] = weights[neighbor] + weight
                    if weights[neighbor] >= nodeThreshold[neighbor]:
                        nodeActived.add(neighbor)
                        newActivitySet.append(neighbor)
        count = count + len(newActivitySet)
        ActivitySet = newActivitySet
    return count


if __name__ == '__main__':
    # Global variables
    n_nodes = 0
    n_edges = 0
    graph = Graph()
    seedlist = []

    #read the arguments from termination
    opts, args = getopt.getopt(sys.argv[1:], 'i:s:m:b:t:r:')
    for (opt, val) in opts:
        if opt == '-i':
            datafile = val
        elif opt == '-s':
            seedfile = val
        elif opt == '-m':
            model_type = val
        elif opt == '-b':
            termination_type = int(val)
        elif opt == '-t':
            runTime = float(val)
        elif opt == '-r':
            random_seed = float(val)


    # datafile = "../test data/NetHEPT.txt"
    # seedfile = "../test data/seeds2.txt"
    # model_type = 'IC'
    # termination_type = 0
    # runTime = 0
    # random_seed = 123
    if model_type == 'IC':
        thismodel = IC
    elif model_type == 'LT':
        thismodel = LT

    read_file(datafile, seedfile)

    # Multiprocess : 7
    q = []
    p = []
    r = 10000
    n = 7
    for i in range(n):
        q.append(Queue())
        p.append(Process(target=ise, args=(q[i], r/n, thismodel, random_seed+i)))
        p[i].start()
    for sub in p:
        sub.join()
    result = []
    for subq in q:
        result.append(subq.get())
    print int(sum(result)/len(result))
