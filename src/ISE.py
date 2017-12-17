from edge import Edge
from graph import Graph
import random
import time
import getopt
import sys


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
        edge = Edge(int(thisline[0]), int(thisline[1]), float(thisline[2]))
        graph.add_edge(edge)

    lines2 = open(seedfile).readlines()
    for i in lines2:
        seedlist.append(int(i))


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
    ActivitySet = seedlist[:]
    nodeActived = set(seedlist)
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

    # Global variables
    n_nodes = 0
    n_edges = 0
    graph = Graph()
    seedlist = []

    # read the arguments from termination
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
    print datafile,seedfile,model_type,termination_type,runTime,random_seed
    quit()


    # datafile = "../test data/NetHEPT.txt"
    # seedfile = "../test data/seeds2.txt"
    # model_type = 'IC'
    # termination_type = 0
    # runTime = 0
    # randomSeed = 123

    random.seed(random_seed)
    read_file(datafile, seedfile)
    print ise(10000, model_type)
    print time.time() - start