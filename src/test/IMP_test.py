from edge import Edge
from graph import Graph
import random
import time
import heapq
from scipy import stats
import numpy as np

# Arguments from commend line
datafile = "../../test data/NetHEPT.txt"
k = 4
model_type = 'IC'
termination_type = 0
runTime = 0
randomSeed = 123

# Global variables
n_nodes = 0
n_edges = 0
graph = Graph()
outdegree = {}
n =0


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


def heuristicsCELF(k, model, model2):
    num_seed = 4*k
    if num_seed > n_nodes:
        num_seed = n_nodes
    seedset = Heuristics3(num_seed, model)
    return model2(k, model, seedset)


def CELF(k, model, seedset):
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

    return S, preSpread



def CELF_improved_11(k, model, seedset):
    # right slow
    global n
    S = set()
    Rs = {100: 3000, 3000: 10000}
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

    return S, preSpread






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

    return S, preSpread

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

    return S, preSpread

def Heuristics3(k, model):
    global outdegree
    h = {}
    S = set()
    # R = 10000
    for node in graph.keys():
        outdegree[node] = graph.outdegree(node)
    for node in graph.keys():
        h[node] = 0
        for e in graph.iteroutedges(node):
            neighbor = e.target
            h[node] += e.weight*outdegree[neighbor]

    for i in range(k):
        winner = max(h, key=h.get)
        h.pop(winner)
        S.add(winner)
        neighbor_winner = graph.neighbor(winner)
        for e in graph.iteroutedges(winner):
            neighbor = e.target
            if neighbor in h:
                union = len(set(neighbor_winner).intersection(set(graph.neighbor(neighbor))))
                h[neighbor] = (1-e.weight)*(h[neighbor]-union)

    # spread = float(0)
    # for i in range(R):
    #     spread = spread + model(S)
    # print spread/R

    #return S, spread/R
    return S

def Heuristics1(k, model):
    global outdegree
    t_dic = {}
    S = set()
    R = 10000
    for node in graph.keys():
        outdegree[node] = graph.outdegree(node)
        t_dic[node] = 0

    outdegree2 = outdegree.copy()
    for i in range(k):
        winner = max(outdegree2, key=outdegree.get)
        outdegree2.pop(winner)
        S.add(winner)
        for e in graph.iteroutedges(winner):
            neighbor = e.target
            if neighbor in outdegree2:
                t_dic[neighbor] += 1
                t = t_dic[neighbor]
                d = outdegree2[neighbor]
                outdegree2[neighbor] = d - 2*t - (d-t)*t*e.weight

    spread = float(0)
    for i in range(R):
        spread = spread + model(S)
    return S, spread/R


def Heuristics2(k, model):
    global outdegree
    h = {}
    S = set()
    R = 10000
    for node in graph.keys():
        outdegree[node] = graph.outdegree(node)
    for node in graph.keys():
        h[node] = 0
        for e in graph.iteroutedges(node):
            neighbor = e.target
            h[node] += e.weight * outdegree[neighbor]

    for i in range(k):
        winner = max(h, key=h.get)
        h.pop(winner)
        S.add(winner)


    spread = float(0)
    for i in range(R):
        spread = spread + model(S)
    return S, spread / R


def ise_IC(seedset):
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
            for edge in graph.iteroutedges(seed):
                neighbor = edge.target
                if neighbor not in nodeActived:
                    if random.random() < edge.weight:
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

    #for k in [1, 4, 10, 20, 30, 50]:
    for k in [1, 5, 8, 10, 50, 100, 200]:
        for model in [ise_IC]:
            #for model2 in [CELF_improved0]:
            #for model2 in [CELF_improved0, CELF_improved, CELF_improved2]:
            for model2 in [CELF_improved_10, CELF_improved21, CELF_improved22]:
                print model2
                t_time = 0
                for i in range(5):
                    start2 = time.time()
                    n = 0
                    # result_g = gernralGreedy(k, model)
                    # result_celf = CELF(k, model, set(graph.keys()))
                    # print "greedy",result_g
                    # print "celf", result_celf
                    # print "Heuristics0",Heuristics0(k, model)
                    # print "Heuristics1", Heuristics1(k, model)
                    # print "Heuristics2", Heuristics2(k, model)
                    #print "Heuristics3", Heuristics3(k, model)
                    print heuristicsCELF(k, model, model2)
                    t_time += time.time()-start2
                    # print result_g[0] == result_celf[0]
                print t_time/5
                print "---------------"
            print "--------------------------------"

    print time.time() - start