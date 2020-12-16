import collections

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score

from fcm import FCM


def clustering(users_skills, n_clusters_range, plot=False):
    X = users_skills

    n_clusters_range = list(n_clusters_range)

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
        kmeans = KMeans(n_clusters=n, random_state=42,
                        n_init=3, max_iter=50).fit(X)
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
            plt.xticks(n_clusters_range)
            plt.ylabel(method_str)
            plt.plot(plotX, Y)
            plt.tight_layout()
            plt.savefig(f"clustering_{method_str}.png")
            plt.close()

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


def fzclustering(users_skills, n_clusters_range, fuzzpar, plot=False):
    X = users_skills
    n_clusters_range = list(n_clusters_range)

    fzmodels = {}

    fpcs = []

    # Find the best number of clusters
    for n_clusters_ in n_clusters_range:
        # another library
        fuzzy_fcm = FCM(n_clusters=n_clusters_, max_iter=50, m=fuzzpar, error=1e-5, random_state=88)
        fuzzy_fcm.fit(X)

        fcm_centers = fuzzy_fcm.centers
        fcm_labels = fuzzy_fcm.predict(X)

        fuzzy_clustering_coeff = fuzzy_fcm.partition_coefficient
        pec = fuzzy_fcm.partition_entropy_coefficient

        fpcs.append(fuzzy_clustering_coeff)

        fzmodels[n_clusters_] = fcm_centers, fcm_labels, fuzzy_clustering_coeff, fuzzy_fcm

    best_centers = max(fzmodels.values(), key=lambda x: x[2])

    if plot:
        plt.figure()
        plt.title(f"Fuzzy c-means over number of clusters")
        plt.xlabel("Number of clusters")
        plt.xticks(n_clusters_range)
        plt.ylabel("Fuzzy partition coefficient")
        plt.plot(n_clusters_range, fpcs)
        plt.tight_layout()
        plt.savefig(f"clustering_Fuzzy_1.png")
        plt.close()

    return best_centers
