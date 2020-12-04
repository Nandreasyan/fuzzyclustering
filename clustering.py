import collections
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score
from fcmeans import FCM


def clustering(users_skills, n_clusters_range, plot=False):
    X = users_skills

    plotX = []

    methods = {
        "silhouette_score": (silhouette_score, 1),
        "davies_bouldin_score": (davies_bouldin_score, -1),
        "calinski_harabasz_score": (calinski_harabasz_score, 1),
    }

    plotY = collections.defaultdict(list)

    # Record the best clustering
    best_score = {k: -np.inf * v[1] for k, v in methods.items()}
    best_n = {k: None for k in methods}

    models = {}

    # Find the best number of clusters
    for n in n_clusters_range:
        plotX.append(n)
        #kmeans = KMeans(n_clusters=n, random_state=42, n_init=3, max_iter=50).fit(X)
        kmeans = FCM(n_clusters=n, random_state=42, max_iter=50).fit(X)

        models[n] = kmeans

        for method_str, (method, optimization_sign) in methods.items():
            score = method(X, kmeans.labels_)

            if score * optimization_sign > best_score[method_str] * optimization_sign:
                best_score[method_str] = score
                best_n[method_str] = len(kmeans.cluster_centers_)

            plotY[method_str].append(score)

    if plot:
        for method_str, Y in plotY.items():
            plt.figure()
            plt.title(f"{method_str} over number of clusters")
            plt.xlabel("Nb clusters")
            plt.ylabel(method_str)
            plt.plot(plotX, Y)
            plt.tight_layout()
            plt.savefig(f"clustering_{method_str}.png")

    # making the metrics vote on a number of cluster to choose
    aggregated_best_n = collections.Counter(best_n.values())
    top_2 = aggregated_best_n.most_common(2)
    print("Votes", top_2)
    # absolute majority
    if len(top_2) < 2:
        best_model = models[top_2[0][0]]
    else:
        # not the same number of vote
        if top_2[0][1] != top_2[1][1]:
            best_model = models[top_2[0][0]]
        else:
            # can't decide, two number of cluster have the same number of metric voting
            return None

    return best_model


def evaluate_clustering(clusters_ground_truth, cluster_predicted):
    print("normalized_mutual_info_score", normalized_mutual_info_score(
        clusters_ground_truth, cluster_predicted))


def fzclustering(users_skills):
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)

    # Set up the loop and plot
    fig1, axes1 = plt.subplots(3, 3, figsize=(10, 10))
    alldata = np.vstack((xpts, ypts))
    fpcs = []

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            users_skills, ncenters, 2, error=0.005, maxiter=1000, init=None)

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

    return fpcs




colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

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

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')

# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
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

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()
