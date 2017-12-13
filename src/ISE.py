from edge import Edge
from graph import Graph
import random
import time

# Arguments from commend line
datafile = "../test data/network.txt"
seedfile = "../test data/seeds.txt"
model = 'IC'
termination_type = 0
runTime = 0
randomSeed = 123

# Global variables
n_nodes = 0
n_edges = 0
graph = Graph()
seedset = []


def read_file(datafile, seedfile):
    """
    Read the network data file and seed data file, to get the graph and seed set.
    :param datafile: the absolute path of network file
    :param seedfile: the absolute path of seed set file
    """
    global n_nodes, n_edges ,graph, seedset
    lines = open(datafile).readlines()
    n_nodes = lines[0].split()[0]
    n_edges = lines[0].split()[1]
    for i in lines[1:]:
        thisline = i.split()
        edge = Edge(int(thisline[0]), int(thisline[1]), float(thisline[2]))
        graph.add_edge(edge)

    lines2 = open(seedfile).readlines()
    for i in lines2:
        seedset.append(int(i))


def ise (times, model):
    '''
    Influence spread estimation
    :param times: the run times
    :param model: The diffusion model: IC or LT
    :return: the average influence spread
    '''
    sum = float(0)
    for i in range(times):
        sum = sum + model()
    return sum/times

def IC():
    '''
    Ise based on Independent Cascade model
    :return: the influence spread
    '''
    ActivitySet = seedset[:]
    nodeActived = set(seedset)
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

def LT():
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
    read_file(datafile, seedfile)
    print n_nodes
    print n_edges
    print seedset

    for model in [IC, LT]:
        print ise(10000, model)
    print time.time() - start