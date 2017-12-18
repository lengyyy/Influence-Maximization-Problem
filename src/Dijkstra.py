
def dijkstra(graph,src):
    # 判断图是否为空，如果为空直接退出
    if graph is None:
        return None
    nodes = [i for i in range(len(graph))]  # 获取图中所有节点
    distance = {}
    path = {}
    forwarding_table = {}

    #fill your code here

    return distance, path, forwarding_table


if __name__ == '__main__':
    graph_list = [[0, 7, float('inf'), 3, 3, 2],
                  [7, 0, 5, float('inf'), 1, 2],
                  [float('inf'), 5, 0, 6, float('inf'), 3],
                  [3, float('inf'), 6, 0, float('inf'), 1],
                  [3, 1, float('inf'), float('inf'), 0, float('inf')],
                  [2, 2, 3, 1, float('inf'), 0]]

    distance, path, forwarding_table = dijkstra(graph_list, 3)  # 查找从源点3开始到其他节点的最短路径
    print(distance)
    print(path)
    print(forwarding_table)