import skfuzzy as fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import dice, cdist
from generate_data import generate_skills, generate_graph

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

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[1, 3],
           [2, 2],
           [3, 8]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.3, 0.5],
          [0.5, 0.3],
          [0.5, 0.3]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))


# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(10, 10))
alldata = np.vstack((xpts, ypts))
fpcs = []



for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    testfuzzy = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc), size=12)
    ax.axis('off')

fig1.tight_layout()

fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs, color='#731810')
ax2.set_title("How Number of Clusters Change FPC?")
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
plt.show()
