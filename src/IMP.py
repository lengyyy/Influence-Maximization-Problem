from edge import Edge
from graph import Graph
import random
import time

# Arguments from commend line
datafile = "../test data/network.txt"
k = 6
model = 'IC'
termination_type = 0
runTime = 0
randomSeed = 123

# Global variables
n_nodes = 0
n_edges = 0
graph = Graph()


def read_file(datafile):
    """
    Read the network data file and seed data file, to get the graph and seed set.
    :param datafile: the absolute path of network file
    :param seedfile: the absolute path of seed set file
    """
    global n_nodes, n_edges, graph
    lines = open(datafile).readlines()
    n_nodes = lines[0].split()[0]
    n_edges = lines[0].split()[1]
    for i in lines[1:]:
        thisline = i.split()
        edge = Edge(int(thisline[0]), int(thisline[1]), float(thisline[2]))
        graph.add_edge(edge)


def imp (model, k):
    '''
    Influence maximization problem
    :param times: the run times
    :param model: The diffusion model: IC or LT
    :param k: The number of seed
    :return: the average influence spread
    '''
    if model == "IC":
        return gernralGreedy_IC(k)
    else:
        pass

def gernralGreedy_IC(k):
    S = set()
    R = 10000
    Candidate = graph.keys()
    for i in range(k):
        addnode = []
        for node in Candidate:
            influenceSpread = float(0)
            newSeed = S.copy()
            newSeed.add(node)
            for i in range(R):
                influenceSpread = influenceSpread + ise_IC(newSeed)
            influenceSpread = influenceSpread/R
            addnode.append((influenceSpread, node))
        addnode.sort(reverse=True)
        print addnode
        winner = addnode[0][1]
        S.add(winner)
        Candidate.remove(winner)
    return S, addnode[0][0]


def ise_IC(seedset):
    '''
    Ise based on Independent Cascade model
    :return: the influence spread
    '''
    ActivitySet = list(seedset)
    nodeActived = seedset.copy()
    count = len(ActivitySet)

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for edge in graph.iteroutedges(seed):
                neighbor = edge.target
                if neighbor not in nodeActived:
                    weight = edge.weight
                    if random.random() < weight:
                        nodeActived.add(neighbor)
                        newActivitySet.append(neighbor)
        count = count + len(newActivitySet)
        ActivitySet = newActivitySet
    return count


def ise_LT(seedset):
    '''
    ISE based on linear threshold model
    :return: the influence spread
    '''
    ActivitySet = seedset[:]
    nodeActived = set(seedset)
    count = len(ActivitySet)
    nodeThreshold = {}
    weights = {}

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for edge in graph.iteroutedges(seed):
                neighbor = edge.target
                if neighbor not in nodeActived:
                    if neighbor not in nodeThreshold:
                        nodeThreshold[neighbor] = random.random()
                        weights[neighbor] = 0
                    weights[neighbor] = weights[neighbor] + edge.weight
                    if weights[neighbor] >= nodeThreshold[neighbor]:
                        nodeActived.add(neighbor)
                        newActivitySet.append(neighbor)
        count = count + len(newActivitySet)
        ActivitySet = newActivitySet
    return count


if __name__ == '__main__':
    start = time.time()
    random.seed()
    read_file(datafile)
    print n_nodes
    print n_edges

    for model in ["IC"]:
        print imp(model, k)
    print time.time() - start