import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from clustering import clustering, evaluate_clustering, fzclustering
from generate_data import skills_gen, generate_graph, skills_gen_fz
from misc import plot_graph

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

# Data generation parameters
skills_sets = [
    ["Assembly", "C", "C++", "Rust"],  # System
    ["Java", "C#", "Go"],  # OOP
    ["JavaScript", "HTML", "CSS", "PHP"],  # Web
    ["Python", "R"],  # Statistics
    ["bash", "zsh", "sh", "batch"],  # Scripting / Shells
    ["SAP", "Microsoft Dynamics", "Odoo", "Spreadsheet"],  # Management
]

seed = int(np.pi * 37)  # Seed for random number generation
np.random.seed(seed)

N = 400  # The number of nodes

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

# Possible distances metrics : "cityblock", "dice", "euclidean", "jaccard", "minkowski"
clustering_range = (2, 10)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Principal component analysis', fontsize=16)
axs[0, 0].set_ylabel('Ground Truth')
axs[1, 0].set_ylabel('KMeans')
axs[1, 1].set_ylabel('Fuzzy CMeans')

def use_case_fuzzy_cmean(users_skills, clusters_ground_truth, fuzzpar):
    print("Clustering")

    fuzzyclustering_model = fzclustering(users_skills, range(*clustering_range), fuzzpar, True)
    # returned values with order
    # Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).
    print("- Number of clusters found", len(fuzzyclustering_model[0]))
    print("- Real number of clusters", len(skills_sets))

    evaluate_clustering(clusters_ground_truth, fuzzyclustering_model[1])

    pca = PCA(n_components=2)
    #
    pca.fit(users_skills)
    new_data = pca.transform(users_skills)
    #
    pca.fit(fuzzyclustering_model[0])
    new_data2 = pca.transform(fuzzyclustering_model[0])
    c = np.concatenate((fuzzyclustering_model[1], np.array([6] * len(fuzzyclustering_model[0]))))
    new_data = np.concatenate((new_data, new_data2), axis=0)
    #
    axs[1, 1].scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)

    # print("Plotting graph")
    plot_graph(G, "Clustered_graph_fuzzy.png", colors=fuzzyclustering_model[1])

    return fuzzyclustering_model


def use_case_kmeans(users_skills, clusters_ground_truth):
    print("Clustering")
    clustering_model = clustering(users_skills, range(*clustering_range), True)
    print("- Number of clusters found", len(clustering_model.cluster_centers_))
    print("- Real number of clusters", len(skills_sets))

    evaluate_clustering(clusters_ground_truth, clustering_model.labels_)

    pca = PCA(n_components=2)
    #
    pca.fit(users_skills)
    new_data = pca.transform(users_skills)
    #
    pca.fit(clustering_model.cluster_centers_)
    new_data2 = pca.transform(clustering_model.cluster_centers_)
    c = np.concatenate((clustering_model.labels_, np.array([6] * len(clustering_model.cluster_centers_))))
    new_data = np.concatenate((new_data, new_data2), axis=0)
    #
    axs[1, 0].scatter(new_data.T[0], new_data.T[1], c=c, alpha=0.5)

    # print("Plotting graph")
    plot_graph(G, "Clustered_graph_K-Means.png", colors=clustering_model.labels_)


if __name__ == '__main__':
    print("Generating skills")
    users_skills, clusters_ground_truth = skills_gen(
        skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

    users_skills_fz, clusters_ground_truth_fz = skills_gen_fz(
        skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

    # Principal component analysis for ground Truth
    pca = PCA(n_components=2)
    pca.fit(users_skills_fz)
    new_data = pca.transform(users_skills_fz)
    axs[0, 0].scatter(new_data.T[0], new_data.T[1].T, c=clusters_ground_truth_fz, alpha=0.5)

    print("Generating graph")
    G = generate_graph(clusters_ground_truth)

    print("Using KMeans")
    use_case_kmeans(users_skills_fz, clusters_ground_truth_fz)

    print("Using Fuzzy C-Means") # third parameter is fuzzification paramater
    test =  use_case_fuzzy_cmean(users_skills_fz, clusters_ground_truth_fz, 1.4)
    inta = 0
    plt.show()
