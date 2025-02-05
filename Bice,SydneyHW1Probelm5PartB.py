"""
Part B: Tune your k and find the number of clusters to achieve a reasonably small mismatch rate.
Please explain how you tune k and what is the achieved mismatch rate. Please explain intuitively what
this result tells about the network community structure
"""

#Same code as Q5 Part A:
import numpy as np
import matplotlib.pyplot as plt

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

# New code: tuning!
#declare k values for tuning
k_values = [2,5,10,20,30,40,50,75,100]
mismatch_rates = np.zeros(len(k_values))

#code from part a again:
for index, k in enumerate(k_values):
    #k eigenvectors of Laplacian
    eigenvalues, eigenvectors=np.linalg.eigh(L)
    V = eigenvectors[:,:k]

    #k-means clustering
    clusters=k_means(V,k)
    clusters=np.array(clusters).flatten()

    #compute majority labels in each cluster
    majority_labels=np.zeros(k,dtype=int)
    for j in range(k):
        cluster_mask=(clusters==j)
        cluster_labels=filtered_labels[cluster_mask]
        if cluster_labels.size > 0:
            majority_labels[j] = np.bincount(cluster_labels).argmax()

    mismatches=np.sum(filtered_labels!=majority_labels[clusters])
    mismatch_rates[index]=mismatches/num_unique_nodes

#find the best k (min mismtach rate)
best_k_index = np.argmin(mismatch_rates)
best_k=k_values[best_k_index]
min_mismatch = mismatch_rates[best_k_index]
print(f'optimal k: {k}')
print(f'optimal k mismatch rate: {min_mismatch:.4f}')

#graph of mismatch rate vs k
plt.figure()
plt.plot(k_values,mismatch_rates,'-o',linewidth=2)
plt.xlabel("Number of Clusters(k)")
plt.ylabel('Mismatch Rate')
plt.title("Tuning k for Specreal Clustering")
plt.grid(True)
plt.show()