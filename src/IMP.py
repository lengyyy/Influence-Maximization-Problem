from edge import Edge
from graph import Graph
import random
import time
import heapq

# Arguments from commend line
datafile = "../test data/network.txt"
k = 4
model_type = 'IC'
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



def gernralGreedy(k, model):
    S = set()
    R = 10000
    Candidate = graph.keys()
    for i in range(k):
        addnode = []
        for node in Candidate:
            Spread = float(0)
            newSeed = S.copy()
            newSeed.add(node)
            for i in range(R):
                Spread = Spread + model(newSeed)
            Spread = Spread/R
            addnode.append((Spread, node))
        addnode.sort(reverse=True)
        winner = addnode[0][1]
        S.add(winner)
        Candidate.remove(winner)
    return S, addnode[0][0]




def CELF(k, model):
    S = set()
    R = 10000
    nodeHeap = []
    preSpread = 1
    for node in graph.keys():
        delta = float(0)
        for i in range(R):
            delta = delta + model({node})
        delta = delta / R - preSpread
        nodeHeap.append((-delta, delta, node, 1))
    heapq.heapify(nodeHeap)
    # while nodeHeap:
    #     print heapq.heappop(nodeHeap)
    # quit()
    winner = heapq.heappop(nodeHeap)
    preSpread = winner[1] + preSpread
    S.add(winner[2])


    for i in range(k-1):
        seedId = i + 2
        while nodeHeap[0][3] != seedId:
            maxOne = nodeHeap[0]
            delta = float(0)
            newSeed = S.copy()
            newSeed.add(maxOne[2])
            for i in range(R):
                delta = delta + model(newSeed)
            delta = delta / R - preSpread
            heapq.heapreplace(nodeHeap,(-delta, delta, maxOne[2], seedId))

        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        S.add(winner[2])

    return S, preSpread


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
    ActivitySet = list(seedset)
    nodeActived = seedset.copy()
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

    print "IC", gernralGreedy(k, ise_IC)
    print "LT", gernralGreedy(k, ise_LT)
    print "CELF_IC", CELF(k, ise_IC)
    print "CELF_LT", CELF(k,ise_LT)
    print time.time() - start