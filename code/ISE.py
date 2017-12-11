from edge import Edge
from graph import Graph
import random
import time

datafile = "../test data/network.txt"
seedfile = "../test data/seeds.txt"
model = 'IC'
termination_type = 0
runTime = 0
randomSeed = 123

n_nodes = 0
n_edges = 0
graph = Graph()
seedset = []
InitNodeStatus = {}


def read_file(datafile, seedfile):
    """

    :param datafile:
    :param seedfile:
    :return:
    """
    global n_nodes, n_edges ,graph, seedset, InitNodeStatus
    lines = open(datafile).readlines()
    n_nodes = lines[0].split()[0]
    n_edges = lines[0].split()[1]
    for i in lines[1:]:
        thisline = i.split()
        edge = Edge(int(thisline[0]), int(thisline[1]), float(thisline[2]))
        graph.add_edge(edge)
    for i in graph.keys():
        InitNodeStatus[i] = 0

    lines2 = open(seedfile).readlines()
    for i in lines2:
        seedset.append(int(i))
        InitNodeStatus[int(i)] = 1


def ise (times, model):
    '''

    :param times:
    :param model:
    :return:
    '''
    sum = float(0)
    if model == "IC":
        for i in range(times):
            sum = sum + IC()
        return sum/times
    else:
        for i in range(times):
            sum = sum + LT()
        return sum/times

def IC():
    '''

    :return:
    '''
    ActivitySet = seedset[:]
    nodeStatus = InitNodeStatus.copy()
    count = len(ActivitySet)

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for edge in graph.iteroutedges(seed):
                neighbor = edge.target
                if nodeStatus[neighbor] == 0:
                    weight = edge.weight
                    if random.random() < weight:
                        nodeStatus[neighbor] = 1
                        newActivitySet.append(neighbor)
        count = count + len(newActivitySet)
        ActivitySet = newActivitySet
    return count

def LT():
    '''

    :return:
    '''
    ActivitySet = seedset[:]
    nodeStatus = InitNodeStatus.copy()
    nodeThreshold = {}
    weights = {}
    for i in graph.keys():
        threshold = random.random()
        nodeThreshold[i] = threshold
        weights[i] = 0
        if threshold == 0:
            print "!"
            ActivitySet.append(i)
            nodeStatus[i] = 1
    count = len(ActivitySet)

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for edge in graph.iteroutedges(seed):
                neighbor = edge.target
                if nodeStatus[neighbor] == 0:
                    weights[neighbor] = weights[neighbor] + edge.weight
                    if weights[neighbor] >= nodeThreshold[neighbor]:
                        nodeStatus[neighbor] = 1
                        newActivitySet.append(neighbor)
        count = count + len(newActivitySet)
        ActivitySet = newActivitySet
    return count



if __name__ == '__main__':
    start = time.time()
    random.seed()
    read_file(datafile, seedfile)
    print n_nodes
    print n_edges
    print seedset

    for model in ["IC","LT"]:
        print ise(10000, model)
    print time.time() - start