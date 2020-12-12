import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import dice

from fcm import FCM
from generate_data import skills_gen, generate_graph

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

N = 100  # The number of nodes

min_skill_sets = 1  # The minimum of skills set to add to a user
max_skill_sets = 2  # The maximal of skills set to add to a user
min_edits = 1  # Mimimum of random edition of the user skill sets
max_edits = 3  # Maximal of random edition of the user skill sets

set_distance_function = dice

print("Generating skills")
users_skills, clusters_ground_truth = skills_gen(
    skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits)

print("Generating graph")
G = generate_graph(clusters_ground_truth)

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]
# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

alldata = np.vstack((xpts, ypts))
fpcs = []

n_samples = 300

X_test = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples, 2)),
    np.random.normal((2, 2), size=(n_samples, 2)),
    np.random.normal((9, 0), size=(n_samples, 2)),
    np.random.normal((5, -8), size=(n_samples, 2))
))

plt.figure(figsize=(5, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], alpha=.1)
plt.show()

n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
for n_clusters in n_clusters_list:
    fcm = FCM(n_clusters)
    fcm.fit(X_test)
    models.append(fcm)

# outputs
num_clusters = len(n_clusters_list)
rows = int(np.ceil(np.sqrt(num_clusters)))
cols = int(np.ceil(num_clusters / rows))
f, axes = plt.subplots(rows, cols, figsize=(11, 16))
for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient

    fcm_centers = model.centers
    fcm_labels = model.predict(X_test)
    # plot result
    axe.scatter(X_test[:, 0], X_test[:, 1], c=fcm_labels, alpha=.1)
    axe.scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=500, c='black')
    axe.set_title(f'n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}')
plt.show()
