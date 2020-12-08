import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from clustering import clustering, evaluate_clustering, fzclustering
from generate_data import generate_skills, generate_graph
from misc import plot_graph

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

seed = int(np.pi * 37)  # Seed for random number generation
np.random.seed(seed)

use_fuzzy_clustering = True

N = 300  # The number of nodes

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

# Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
clustering_range = (2, 10)
distance_function = "euclidean"


def use_case_fuzzy_cmean(users_skills, clusters_ground_truth):
    print("Clustering")
    fuzzyclustering_model = fzclustering(users_skills, range(*clustering_range), True)
    # returned values with order
    # Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).
    print("- Number of clusters found", len(fuzzyclustering_model[0]))
    print("- Real number of clusters", len(skills_sets))

    users_distances_to_centers = cdist(
        users_skills, fuzzyclustering_model[0], metric=distance_function)
    print(" Evluate clsutering fuzzy ", evaluate_clustering(clusters_ground_truth, fuzzyclustering_model[1]))

    pca = PCA(n_components=2)

    pca.fit(users_skills)
    new_data = pca.transform(users_skills)

    pca.fit(fuzzyclustering_model[0])
    new_data2 = pca.transform(fuzzyclustering_model[0])
    c = np.concatenate((fuzzyclustering_model[1], np.array([6] * 6)))
    new_data = np.concatenate((new_data, new_data2), axis=0)

    plt.scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)
    plt.show()


    # print("Plotting graph")
    plot_graph(G, "Clustered_graph_fuzzy.png", colors=fuzzyclustering_model[1])


def use_case_kmeans(users_skills, clusters_ground_truth):
    print("Clustering")
    clustering_model = clustering(users_skills, range(*clustering_range), True)
    print("- Number of clusters found", len(clustering_model.cluster_centers_))
    print("- Real number of clusters", len(skills_sets))

    users_distances_to_centers = cdist(
        users_skills, clustering_model.cluster_centers_, metric=distance_function)
    print("Kmenas mujtual score", evaluate_clustering(clusters_ground_truth, clustering_model.labels_))

    # pca = PCA(n_components=2)
    #
    # pca.fit(users_skills)
    # new_data = pca.transform(users_skills)
    #
    # pca.fit(clustering_model.cluster_centers_)
    # new_data2 = pca.transform(clustering_model.cluster_centers_)
    # c = np.concatenate((clustering_model.labels_, np.array([6] * 6)))
    # new_data = np.concatenate((new_data, new_data2), axis=0)
    #
    # plt.scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)
    # plt.show()

    # print("Plotting graph")
    plot_graph(G, "Clustered_graph_K-Means.png", colors=clustering_model.labels_)


if __name__ == '__main__':
    print("Generating skills")
    users_skills, clusters_ground_truth = generate_skills(
        skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

    # pca = PCA(n_components=2)
    # pca.fit(users_skills)
    # new_data = pca.transform(users_skills)
    # plt.scatter(new_data.T[0], new_data.T[1].T,  c=clusters_ground_truth, alpha=0.5)
    # plt.show()

    # print('pcaexplaned userskills', pca.explained_variance_ratio_)
    # print('singular values', pca.singular_values_)


    print("Generating graph")
    G = generate_graph(clusters_ground_truth)

    # print("Using KMeans")
    #use_case_kmeans(users_skills, clusters_ground_truth)

    # print("Using Fuzzy C-Means")
    use_case_fuzzy_cmean(users_skills, clusters_ground_truth)
