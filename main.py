import numpy as np
from scipy.spatial.distance import dice, cdist

from generate_data import generate_skills, generate_graph
from clustering import clustering, evaluate_clustering, fzclustering
from misc import plot_graph
from fcmeans import FCM

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

# Data generation parameters
skills_sets = [
    ["Assembly", "C", "C++", "Rust"],  # System
    ["Java", "C#", "Go"],  # OOP
    ["Python", "R"],  # Statistics
    ["bash", "zsh", "sh", "batch"],  # Scripting / Shells
    ["JavaScript", "HTML", "CSS", "PHP"],  # Web
    ["SAP", "Microsoft Dynamics", "Odoo", "Spreadsheet"],  # Management
]

seed = int(np.pi * 42)  # Seed for random number generation
np.random.seed(seed)

N = 200  # The number of nodes
K = 4  # Each node is connected to k nearest neighbors in ring topology
P = 0.2  # The probability of rewiring each edge

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

set_distance_function = dice

print("Generating skills")
users_skills, clusters_ground_truth = generate_skills(
    skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

print("Generating graph")
G = generate_graph(N, K, P, seed)

print("Clustering")
clustering_model = clustering(users_skills, range(2, 10), True)
fzclustering_model = fzclustering(users_skills)
# Possible distances metrics : "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule".
users_distances_to_centers = cdist(
    users_skills, clustering_model.cluster_centers_, metric="minkowski")
nb_clusters_found = len(clustering_model.cluster_centers_)

#fzclustering
users_distances_to_centers_fz = cdist(
    users_skills, fzclustering_model.cluster_centers_, metric="minkowski")
nb_clusters_found_fz = len(fzclustering_model.cluster_centers_)


print("Number of clusters found FZ", nb_clusters_found)
print("Real number of clusters FZ", len(skills_sets))


print("Number of clusters found", nb_clusters_found_fz)
print("Real number of clusters", len(skills_sets))

evaluate_clustering(clusters_ground_truth, clustering_model.labels_)

# print("Plotting graph")
plot_graph(G, "graph2.png", colors=clustering_model.labels_)

# print("Link prediction")
# link_prediction_model = link_prediction(G, users_distances_to_centers)

# predictions = predict_links(link_prediction_model,
#                            G, 0, users_distances_to_centers)
# print(predictions)
