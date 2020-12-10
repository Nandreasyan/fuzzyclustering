from fcmeans import FCM
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial.distance import dice, cdist

from generate_data import skills_gen, generate_graph
from clustering import clustering, evaluate_clustering
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
users_skills, clusters_ground_truth = skills_gen(
    skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

print("Generating graph")
G = generate_graph(N, K, P, seed)


n_samples = 3000

X = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples, 2)),
    np.random.normal((2, 2), size=(n_samples, 2))
))

# fit the fuzzy-c-means
fz_kmeans = FCM(n_clusters=3)
fz_kmeans.fit(X)

# outputs
fcm_centers = fz_kmeans.centers
fcm_labels = fz_kmeans.u.argmax(axis=1)


# plot result
f, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].scatter(users_skills[:, 0], users_skills[:, 1], alpha=.1)
axes[1].scatter(users_skills[:, 0], users_skills[:, 1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="s", s=100, c='white')
plt.show()