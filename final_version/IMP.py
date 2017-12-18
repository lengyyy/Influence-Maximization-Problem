import random
import time
import heapq
from scipy import stats
import numpy as np
import sys
import getopt
from multiprocessing import Process, Queue, TimeoutError
import copy
import signal


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)




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


def CELF(k, seedset):##
    global p, q_in, q_out
    global n
    S = set()
    nodeHeap = []
    preSpread = 0
    for node in seedset:
        for qin in q_in:
            qin.put(True)
            qin.put(10000/7)
            qin.put({node})
            qin.put(preSpread)
        result = []
        for qout in q_out:
            result.append(qout.get(True))
        high = sum(result) / len(result)
        nodeHeap.append((-high, high, node, 1))
    heapq.heapify(nodeHeap)
    winner = heapq.heappop(nodeHeap)
    preSpread = winner[1] + preSpread
    S.add(winner[2])

    for i1 in range(k-1):
        seedId = i1 + 2
        while nodeHeap[0][3] != seedId:
            maxOne = nodeHeap[0]
            newSeed = S.copy()
            newSeed.add(maxOne[2])
            for qin in q_in:
                qin.put(True)
                qin.put(10000 / 7)
                qin.put(newSeed)
                qin.put(preSpread)
            result = []
            for qout in q_out:
                result.append(qout.get(True))
            delta = sum(result) / len(result)
            heapq.heapreplace(nodeHeap, (-delta, delta, maxOne[2], seedId))


        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        S.add(winner[2])

    return S


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
    h = sorted(h.items(), key=lambda e: e[1], reverse=True)
    for i in range(k):
        S.add(h[i][0])
    return S


def Heuristics_improved(k):
    outdegree = {}
    h = {}
    S = set()
    for node in res_graph.keys():
        outdegree[node] = res_graph.outdegree(node)
    for node in res_graph.keys():
        h[node] = 0
        for neighbor, weight in res_graph.neighbor(node):
            h[node] += weight*outdegree[neighbor]

    for i in range(k):
        winner = max(h, key=h.get)
        h.pop(winner)
        S.add(winner)
        for neighbor, weight in res_graph.neighbor(winner):
               if neighbor in h:
                union = len(set(res_graph.neighbor_node(winner)).intersection(set(graph.neighbor_node(neighbor))))
                h[neighbor] = (1-weight)*(h[neighbor]-union)
    return S


def heuristics_CELF_improved(k):
    num_seed = 8*k
    if num_seed > n_nodes:
        num_seed = n_nodes
    seedset = Heuristics(num_seed)
    return CELF_improved(k, seedset)


def CELF_improved(k, seedset):
    global p, q_in, q_out, final_seed
    Rs = {1000: 10000}
    nodeHeap = []
    preSpread = 0
    for node in seedset:
        for qin in q_in:
            qin.put(False)
            qin.put(1000/7)
            qin.put({node})
            qin.put(preSpread)
        result = []
        for qout in q_out:
            result.append(qout.get(True))
        high = sum(result) / len(result)
        nodeHeap.append((-high, high, node, -1, 100))
    heapq.heapify(nodeHeap)

    for i1 in range(k):

        while nodeHeap[0][3] != i1 or nodeHeap[0][4] != 10000:
            maxOne = nodeHeap[0]
            newSeed = final_seed.copy()
            newSeed.add(maxOne[2])
            if maxOne[3] == i1:
                thisR = Rs[maxOne[4]]
            else:
                thisR = 1000

            if thisR == 10000:
                for qin in q_in:
                    qin.put(True)
                    qin.put(10000 / 7)
                    qin.put(newSeed)
                    qin.put(preSpread)
                result = []
                for qout in q_out:
                    result.append(qout.get(True))
                delta = sum(result) / len(result)
                heapq.heapreplace(nodeHeap, (-delta, delta, maxOne[2], i1, thisR))
            else:
                for qin in q_in:
                    qin.put(False)
                    qin.put(thisR / 7)
                    qin.put(newSeed)
                    qin.put(preSpread)
                result = []
                for qout in q_out:
                    result.append(qout.get(True))
                high = sum(result) / len(result)
                heapq.heapreplace(nodeHeap, (-high, high, maxOne[2], i1, thisR))

        winner = heapq.heappop(nodeHeap)
        preSpread = winner[1] + preSpread
        final_seed.add(winner[2])





def ise(random_seed, model, q_in, q_out):
    '''
    Influence spread estimation
    :param times: the run times
    :param model: The diffusion model: IC or LT
    :return: the average influence spread
    '''
    random.seed(random_seed)
    while True:
        Accurate = q_in.get(True)
        times = q_in.get(True)
        seedset = q_in.get(True)
        preSpread = q_in.get(True)

        if Accurate is True:
            tem = []
            for i in range(times):
                tem.append(model(seedset))
            q_out.put(float(sum(tem)) / times - preSpread)
        else:
            delta = []
            for i in range(times):
                delta.append(model(seedset) - preSpread)
            std = stats.sem(delta)
            if std == 0:
                high = delta[0]
            else:
                high = stats.t.interval(0.95, len(delta) - 1, loc=np.mean(delta), scale=std)[1]
            q_out.put(high)


def ise_finalresult(model, seedset):
    '''
    Influence spread estimation
    :param times: the run times
    :param model: The diffusion model: IC or LT
    :return: the average influence spread
    '''
    tem = []
    for i in range(10000):
        tem.append(model(seedset))
    return float(sum(tem)) / 10000



def IC(seedset):
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
    # Global variables
    n_nodes = 0
    n_edges = 0
    graph = Graph()
    final_seed = set()

    # read the arguments from termination
    opts, args = getopt.getopt(sys.argv[1:], 'i:k:m:b:t:r:')
    for (opt, val) in opts:
        if opt == '-i':
            datafile = val
        elif opt == '-k':
            k = int(val)
        elif opt == '-m':
            model_type = val
        elif opt == '-b':
            termination_type = int(val)
        elif opt == '-t':
            runTime = int(val)
        elif opt == '-r':
            random_seed = float(val)

    # datafile = "../test data/network.txt"
    # k = 4
    # model_type = 'LT'
    # termination_type = 0
    # runTime = 1
    # random_seed = 123

    if model_type == 'IC':
        thismodel = IC
    elif model_type == 'LT':
        thismodel = LT
    random.seed(random_seed)
    read_file(datafile)

    q_in = []
    q_out = []
    p = []
    n = 7
    for i in range(n):
        q_in.append(Queue())
        q_out.append(Queue())
        p.append(Process(target=ise, args=(random_seed+i, thismodel, q_in[i], q_out[i])))
        p[i].start()

    if termination_type == 1:
        try:
            with timeout(seconds=runTime - 1):
                heuristics_CELF_improved(k)
        except TimeoutError:
            pass
        finally:
            for sub in p:
                sub.terminate()
    elif termination_type == 0:
        heuristics_CELF_improved(k)
        for sub in p:
            sub.terminate()

    # If not enough time
    res = k-len(final_seed)
    if res != 0:
        res_graph = copy.deepcopy(graph)
        for node in final_seed:
            res_graph.del_node(node)
        res_seed = Heuristics_improved(res)
        for s in final_seed:
            print s
        for s in res_seed:
            print s
    else:
        for s in final_seed:
            print s

