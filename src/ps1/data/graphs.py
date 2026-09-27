from collections import deque
import numpy as np


def local_groups(adjacency, maximum_neighbors=4):
    """Sensor plus nearest reachable nodes; ID breaks equal-hop ties."""
    adjacency=np.asarray(adjacency)
    if adjacency.ndim!=2 or adjacency.shape[0]!=adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    groups=[]
    for source in range(len(adjacency)):
        distance={source:0}; queue=deque([source])
        while queue:
            node=queue.popleft()
            for neighbor in np.flatnonzero(adjacency[node]>0):
                neighbor=int(neighbor)
                if neighbor not in distance:
                    distance[neighbor]=distance[node]+1; queue.append(neighbor)
        ordered=sorted((d,node) for node,d in distance.items() if node!=source)
        groups.append(tuple([source]+[node for _,node in ordered[:maximum_neighbors]]))
    return tuple(groups)
