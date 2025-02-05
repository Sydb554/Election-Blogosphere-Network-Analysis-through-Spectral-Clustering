"""
We will study a political blog dataset first compiled for the paper Lada A. Adamic and Natalie Glance,
“The political blogosphere and the 2004 US Election”, in Proceedings of the WWW-2005 Workshop on the
Weblogging Ecosystem (2005). It is assumed that blog-site with the same political orientation are more
likely to link to each other, thus, forming a “community” or “cluster” in a graph. In this question, we will
see whether or not this hypothesis is likely to be true based on the data.
• The dataset nodes.txt contains a graph with n = 1490 vertices (“nodes”) corresponding to political
blogs.
• The dataset edges.txt contains edges between the vertices. You may remove isolated nodes (nodes that
are not connected to any other nodes) in the pre-processing.
We will treat the network as an undirected graph; thus, when constructing the adjacency matrix, make
it symmetrical by, e.g., set the entry in the adjacency matrix to be one whether there is an edge between
the two nodes (in either direction).
In addition, each vertex has a 0-1 label (in the 3rd column of the data file) corresponding to the true
political orientation of that blog. We will consider this as the true label and check whether spectral clustering
will cluster nodes with the same political orientation as possible.

Part A: Use spectral clustering to find the k = 2, 5, 10, 25 clusters in the network of political blogs
(each node is a blog, and their edges are defined in the file edges.txt). Find majority labels in each
cluster for different k values, respectively. For example, if there are k = 2 clusters, and their labels are
{0, 1, 1, 1} and {0, 0, 1} then the majority label for the first cluster is 1 and for the second cluster is 0.
It is required you implement the algorithms yourself rather than calling from a package.

Now compare the majority label with the individual labels in each cluster, and report the mismatch
rate for each cluster, when k = 2, 5, 10, 25. For instance, in the example above, the mismatch rate for
the first cluster is 1/4 (only the first node differs from the majority), and the second cluster is 1/3.
"""

import numpy as np

#Step 1. Open file and load nodes and edges into variables for use
#columns having problems so loading data manually
nodes_data=[]
with open('nodes.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 4:  #missing columns: need to add placeholder to fill them in
            parts.extend([''] * (4 - len(parts))) 
        nodes_data.append(parts)

#fill nodes data array
nodes_data = np.array(nodes_data,dtype=object)
node_ids=nodes_data[:,0].astype(int)

#for missing label on that one row of data
labels=np.array([int(x) if x.isdigit() else 0 for x in nodes_data[:,2]])

edges_data=np.loadtxt('edges.txt',dtype=int,delimiter='\t')

#Step 2: Map node ID to index
unique_nodes, index_map =np.unique(edges_data,return_inverse=True)
num_unique_nodes=len(unique_nodes)
node_map= {node:i for i, node in enumerate(unique_nodes)}

#Step 3: Adjacency matrix
A = np.zeros((num_unique_nodes,num_unique_nodes), dtype=int)
for i in range(edges_data.shape[0]):
    n1,n2 = edges_data[i]
    if n1 in node_map and n2 in node_map:
        idx1,idx2=node_map[n1],node_map[n2]
        A[idx1,idx2]=1
        A[idx2,idx1]=1

#Can only use labels for available nodes
filtered_labels=labels[np.isin(node_ids,unique_nodes)]
filtered_labels=np.array(filtered_labels).flatten()

#Step 4: Degree matrix
D=np.diag(A.sum(axis=1))

#Step 5: Laplacian matrix (L=D-A)
L=D-A

#k-values for clustering
k_values=[2,5,10,30,50]

#Step 6: function to get the k-means (slightly altered from Q3 and Q4)
def k_means(pixels, k, max_iter = 100, tolerance=0.001):
    np.random.seed(42)
    #Step 2: initialize random centroids
    centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
    
    for iter in range(max_iter):
        distances = np.linalg.norm(pixels[:,None]-centroids,axis=2)
        labels = np.argmin(distances,axis=1)
        new_centroids=np.array([pixels[labels ==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.all(centroids==new_centroids):
            break
        centroids= new_centroids
    return labels

for k in k_values:
    #k eigenvectors of Laplacian
    eigenvalues, eigenvectors=np.linalg.eigh(L)
    V = eigenvectors[:,:k]

    #k-means clustering
    clusters=k_means(V,k)
    clusters=np.array(clusters).flatten()

    #mismatch rates
    mismatch_rates=np.zeros(k,dtype=float)

    #calculate overall mismatch rate
    total_mismatches=0
    total_elements =0

    #compute majority labels in each cluster
    majority_labels=np.zeros(k,dtype=int)
    for i in range(k):
        cluster_mask=(clusters==i)
        cluster_labels = filtered_labels[cluster_mask]
        #for not empty clisters
        if cluster_labels.size > 0:
            majority_label = np.bincount(cluster_labels).argmax()  # Get majority label
            majority_count = np.bincount(cluster_labels)[majority_label]
            mismatch_count = cluster_labels.size - majority_count
            mismatch_rate = mismatch_count / cluster_labels.size  # Calculate mismatch rate
            majority_labels[i] = majority_label
            mismatch_rates[i] = mismatch_rate
            total_mismatches += mismatch_count
            total_elements += cluster_labels.size
    print(f'For k = {k}, majority labels per cluster:')
    print(majority_labels)
    print(mismatch_rates)

    overall_mismatch_rate = total_mismatches/total_elements if total_elements >0 else 0
    print(f'Overall mismatch rate: {overall_mismatch_rate}')