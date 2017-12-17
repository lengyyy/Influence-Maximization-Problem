from graph import Graph
import random
import time
import heapq
from scipy import stats
import numpy as np
import sys
import getopt


def read_file(datafile):
    """
    Read the network data file and seed data file, to get the graph and seed set.
    :param datafile: the absolute path of network file
    :param seedfile: the absolute path of seed set file
    """
    global n_nodes, n_edges, graph
    lines = open(datafile).readlines()
    n_nodes = int(lines[0].split()[0])
    n_edges = int(lines[0].split()[1])
    for i in lines[1:]:
        thisline = i.split()
        graph.add_edge(int(thisline[0]), int(thisline[1]), float(thisline[2]))


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


def CELF(k, model, seedset):##
    seedset = graph.keys()
    global n
    S = set()
    R = 10000
    nodeHeap = []
    preSpread = 0
    for node in seedset:
        delta = float(0)
        for i in range(R):
            delta = delta + model({node})
        delta = delta / R
        nodeHeap.append((-delta, delta, node, 1))
    heapq.heapify(nodeHeap)
    winner = heapq.heappop(nodeHeap)
    preSpread = winner[1] + preSpread
    S.add(winner[2])

    for i1 in range(k-1):
        seedId = i1 + 2
        while nodeHeap[0][3] != seedId:
            #print seedId
            maxOne = nodeHeap[0]
            delta = float(0)
            newSeed = S.copy()
            newSeed.add(maxOne[2])
            for i in range(R):
                delta = delta + model(newSeed)
            delta = delta / R - preSpread
            heapq.heapreplace(nodeHeap, (-delta, delta, maxOne[2], seedId))

        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        S.add(winner[2])

    return S, preSpread


def Heuristics(k):
    outdegree = {}
    h = {}
    S = set()
    for node in graph.keys():
        outdegree[node] = graph.outdegree(node)
    for node in graph.keys():
        h[node] = 0
        for neighbor, weight in graph.neighbor(node):
            h[node] += weight*outdegree[neighbor]

    for i in range(k):
        winner = max(h, key=h.get)
        h.pop(winner)
        S.add(winner)
        for neighbor, weight in graph.neighbor(winner):
            if neighbor in h:
                union = len(set(graph.neighbor_node(winner)).intersection(set(graph.neighbor_node(neighbor))))
                h[neighbor] = (1-weight)*(h[neighbor]-union)

    return S


def heuristics_CELF_improved(k, model, model2):
    num_seed = 4*k
    if num_seed > n_nodes:
        num_seed = n_nodes
    seedset = Heuristics(num_seed)
    return model2(k, model, seedset)


def CELF_improved_10(k, model, seedset):
    global n
    S = set()
    Rs = {100: 1000, 1000: 10000}
    nodeHeap = []
    preSpread = 0
    for node in seedset:
        delta = []
        for i in range(100):
            delta.append(model({node}))
        std = stats.sem(delta)
        if std == 0:
            high = delta[0]
        else:
            high = stats.t.interval(0.95, len(delta) - 1, loc=np.mean(delta), scale=std)[1]
        nodeHeap.append((-high, high, node, -1, 100))
    heapq.heapify(nodeHeap)

    for i1 in range(k):

        while nodeHeap[0][3] != i1 or nodeHeap[0][4] != 10000:
            maxOne = nodeHeap[0]
            newSeed = S.copy()
            newSeed.add(maxOne[2])
            if maxOne[3] == i1:
                thisR = Rs[maxOne[4]]
            else:
                thisR = 100

            if thisR == 10000:
                delta = float(0)
                for i in range(thisR):
                    delta = delta + model(newSeed)
                delta = delta / thisR - preSpread
                heapq.heapreplace(nodeHeap, (-delta, delta, maxOne[2], i1, thisR))
            else:
                deltas = []
                for i in range(thisR):
                    deltas.append(model(newSeed)-preSpread)
                std = stats.sem(deltas)
                if std == 0:
                    high = deltas[0]
                else:
                    high = stats.t.interval(0.95, len(deltas) - 1, loc=np.mean(deltas), scale=std)[1]
                heapq.heapreplace(nodeHeap, (-high, high, maxOne[2], i1, thisR))

        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        S.add(winner[2])

    return S


def CELF_improved21(k, model, seedset):
    # direct celf
    global n
    S = set()
    R = 10000
    nodeHeap = []
    preSpread = 0
    for node in seedset:
        delta = []
        for i in range(500):
            delta.append(model({node}))
        std = stats.sem(delta)
        if std == 0:
            high = delta[0]
        else:
            high = stats.t.interval(0.95, len(delta) - 1, loc=np.mean(delta), scale=std)[1]
        nodeHeap.append((-high, high, node, -1))
    heapq.heapify(nodeHeap)

    for i1 in range(k):
        while nodeHeap[0][3] != i1:
            #print seedId
            maxOne = nodeHeap[0]
            delta = float(0)
            newSeed = S.copy()
            newSeed.add(maxOne[2])
            for i in range(R):
                delta = delta + model(newSeed)
            delta = delta / R - preSpread
            heapq.heapreplace(nodeHeap, (-delta, delta, maxOne[2], i1))

        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        S.add(winner[2])

    return S


def CELF_improved22(k, model, seedset):
    # direct celf
    global n
    S = set()
    R = 10000
    nodeHeap = []
    preSpread = 0
    for node in seedset:
        delta = []
        for i in range(1000):
            delta.append(model({node}))
        std = stats.sem(delta)
        if std == 0:
            high = delta[0]
        else:
            high = stats.t.interval(0.95, len(delta) - 1, loc=np.mean(delta), scale=std)[1]
        nodeHeap.append((-high, high, node, -1))
    heapq.heapify(nodeHeap)

    for i1 in range(k):
        while nodeHeap[0][3] != i1:
            #print seedId
            maxOne = nodeHeap[0]
            delta = float(0)
            newSeed = S.copy()
            newSeed.add(maxOne[2])
            for i in range(R):
                delta = delta + model(newSeed)
            delta = delta / R - preSpread
            heapq.heapreplace(nodeHeap, (-delta, delta, maxOne[2], i1))

        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        S.add(winner[2])

    return S


def ise(times, model, seedset, preSpread=0):
    '''
    Influence spread estimation
    :param times: the run times
    :param model: The diffusion model: IC or LT
    :return: the average influence spread
    '''
    if times == 10000:
        sum = float(0)
        for i in range(times):
            sum = sum + model(seedset)
        return sum/times - preSpread
    else:
        delta = []
        for i in range(times):
            delta.append(model(seedset) - preSpread)
        return delta



def IC(seedset):
    '''
    Ise based on Independent Cascade model
    :return: the influence spread
    '''
    global n
    n += 1
    ActivitySet = list(seedset)
    nodeActived = seedset.copy()
    count = len(ActivitySet)

    while ActivitySet:
        newActivitySet = []
        for seed in ActivitySet:
            for neighbor, weight in graph.neighbor(seed):
                if neighbor not in nodeActived:
                    if random.random() < weight:
                        nodeActived.add(neighbor)
                        newActivitySet.append(neighbor)
        count = count + len(newActivitySet)
        ActivitySet = newActivitySet
    return count


def LT(seedset):
    '''
    ISE based on linear threshold model
    :return: the influence spread
    '''
    global n
    n += 1
    ActivitySet = list(seedset)
    nodeActived = seedset.copy()
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
    n = 0##

    # read the arguments from termination
    # opts, args = getopt.getopt(sys.argv[1:], 'i:k:m:b:t:r:')
    # for (opt, val) in opts:
    #     if opt == '-i':
    #         datafile = val
    #     elif opt == '-k':
    #         k = int(val)
    #     elif opt == '-m':
    #         model_type = val
    #     elif opt == '-b':
    #         termination_type = int(val)
    #     elif opt == '-t':
    #         runTime = float(val)
    #     elif opt == '-r':
    #         random_seed = float(val)

    datafile = "../test data/network.txt"
    k = 10
    model_type = 'LT'
    termination_type = 0
    runTime = 0
    random_seed = 123

    random.seed(random_seed)
    if model_type == 'IC':
        thismodel = IC
    elif model_type == 'LT':
        thismodel = LT
    read_file(datafile)

    result = heuristics_CELF_improved(k, thismodel, CELF_improved_10)
    # result_alt = CELF(k, IC)


    # If not enough time
    res = k-len(result)
    if res != 0:
        print "not enough!"##
        for node in result:
            graph.del_node(node)
        res_seed = Heuristics(res)
        for s in result:
            print s
        for s in res_seed:
            print s
        print ise(10000, thismodel, result+res_seed)##
    else:
        for s in result:
            print s
        print ise(10000, thismodel, result)##

    print time.time() - start