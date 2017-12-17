from graph import Graph
import random
import time
import numpy as np
import getopt
import sys
from multiprocessing import Process, Queue


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
    q.put(float(sum(tem))/times)

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
    start = time.time()

    # Global variables
    n_nodes = 0
    n_edges = 0
    graph = Graph()
    seedlist = []

    # read the arguments from termination
    # opts, args = getopt.getopt(sys.argv[1:], 'i:s:m:b:t:r:')
    # for (opt, val) in opts:
    #     if opt == '-i':
    #         datafile = val
    #     elif opt == '-s':
    #         seedfile = val
    #     elif opt == '-m':
    #         model_type = val
    #     elif opt == '-b':
    #         termination_type = int(val)
    #     elif opt == '-t':
    #         runTime = float(val)
    #     elif opt == '-r':
    #         random_seed = float(val)


    datafile = "../test data/NetHEPT.txt"
    seedfile = "../test data/seeds2.txt"
    model_type = 'IC'
    termination_type = 0
    runTime = 0
    random_seed = 123

    if model_type == 'IC':
        thismodel = IC
    elif model_type == 'LT':
        thismodel = LT
    read_file(datafile, seedfile)

    q = []
    p = []
    r = 10000
    n = 7
    print time.time()-start
    for i in range(n):
        q.append(Queue())
        p.append(Process(target=ise, args=(q[i], r/n, thismodel, random_seed+i)))
        p[i].start()

    for sub in p:
        sub.join()


    result = []
    for subq in q:
        result.append(subq.get())
    print result
    print sum(result)/len(result)


    print time.time() - start