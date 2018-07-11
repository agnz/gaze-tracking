import numpy as np
import scipy
from scipy.spatial.distance import euclidean as euclid_dist


def _chinese_whispers(encodings, tubelet_overlap, threshold=0.4, iterations=20):
    """ Chinese Whispers Algorithm
    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate
    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    from random import shuffle
    import networkx as nx

    def draw_G(pos=[]):
        # color by cluster
        import matplotlib.pyplot as plt
        _color = [-1] * len(G)
        # nx.get_node_attributes(G,'cluster')
        for node_id, node in G.node.items():
            if 'cluster' in node:
                _color[node_id-1] = colors[node['cluster']-1]
        if not pos:
            pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=_color, cmap=plt.get_cmap('hsv'))
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        plt.show()
        return pos


    # Create graph
    nodes = []
    edges = []

    if len(encodings) <= 1:
        print("Not enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Graph node ids start from 1
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': idx, 'original_idx': idx})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        distances = np.squeeze(scipy.spatial.distance.cdist(compare_encodings, [face_encoding_to_check], 'Euclidean'),
                               axis=1)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance < threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                if tubelet_overlap[node_id-1, edge_id-1]:
                    encoding_edges.append((node_id, edge_id, {'weight': -100}))
                else:
                    encoding_edges.append((node_id, edge_id, {'weight': 1/distance}))
            else:
                edge_id = idx + i + 2
                if tubelet_overlap[node_id-1, edge_id-1]:
                    encoding_edges.append((node_id, edge_id, {'weight': -100}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # colors = list(range(len(G)))
    # colors = [color / len(G) for color in colors]
    # shuffle(colors)
    # pos = draw_G()

    # [node for node,cluster in nx.get_node_attributes(G,'cluster').items() if cluster==13]
    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = list(G)
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = G.node[node]['cluster']
            #use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}


    # Prepare cluster output
    for (node_id, data) in G.node.items():
        cluster = data['cluster']
        original_idx = data['original_idx']

        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(original_idx)

    # Sort cluster output
    sorted_clusters = sorted(list(clusters.values()), key=len, reverse=True)

    # draw_G(pos)

    return sorted_clusters

# def kmeans():
    # # kmeans clustering
    # dist = scipy.spatial.distance.cdist(tub_mean_descriptor, tub_mean_descriptor, 'Euclidean')
    # # kmeans = KMeans(n_clusters=3, precompute_distances=True).fit(dist)
    # kmeans = KMeans(n_clusters=3).fit_predict(tub_mean_descriptor)
    #
    # names_ = []
    # for ll in kmeans:
    #     names_.append(names.get_first_name(gender='female'))
    #
    # # new_labels = np.ones(len(tubelets),dtype=int)*-1
    # new_labels = [' '] * len(tubelets)
    # for n, ii in enumerate(included):
    #     new_labels[ii] = names_[kmeans[n]]
    #
    # for k in range(dataset['num_frames']):
    #     cur_frame_dets = data[k]
    #     for det in cur_frame_dets:
    #         # det['label'] = kmeans[det['label']]
    #         det['label'] = new_labels[det['label']]
    #
    # for n, t in enumerate(tubelets):
    #     print("tublet {:2d}, label: {}, frames: {:5d}".format(n, new_labels[n],len(t)))
    # # print(kmeans.labels_)
