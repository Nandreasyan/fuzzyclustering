import numpy as np
import networkx as nx


def generate_skills(skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits):
    """Generate N random users_skills based on skills_sets
    add a bit of noise with the min and max edits
    """
    all_skills = list()
    for ss in skills_sets:
        all_skills += ss

    users_skills = []
    clusters_ground_truth = []

    for _ in range(N):
        user_skills = np.zeros((len(all_skills)), dtype=bool)

        nb_skill_sets = np.random.randint(min_skill_sets, max_skill_sets)
        skills_sets_indices = np.random.choice(
            range(len(skills_sets)), nb_skill_sets)

        for s in skills_sets_indices:
            skill_set = skills_sets[s]
            for skill in skill_set:
                i = all_skills.index(skill)
                user_skills[i] = True

        nb_edits = np.random.randint(min_edits, max_edits)
        for _ in range(nb_edits):
            # flip a random bit
            nbTrue = user_skills.sum()
            nbFalse = len(user_skills) - nbTrue

            a = np.zeros((len(user_skills)))
            a[user_skills] = nbTrue
            a[np.logical_not(user_skills)] = nbFalse
            p = np.full((len(user_skills)), 0.5) / a

            i = np.random.choice(range(len(all_skills)), p=p)
            user_skills[i] ^= True

        users_skills.append(user_skills)
        clusters_ground_truth.append(skills_sets_indices[0])

    users_skills = np.array(users_skills)
    clusters_ground_truth = np.array(clusters_ground_truth)

    return users_skills, clusters_ground_truth


def generate_graph(clusters_ground_truth, cluster_boost=3, m=2):
    """Creating a graph according to the PREFERENTIAL ATTACHMENT MODEL
    for a social graph alike"""
    G = nx.Graph()

    # initialize the two first users
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1)

    for c_node, cluster in list(enumerate(clusters_ground_truth))[2:]:
        candidates = list(G.nodes)
        G.add_node(c_node)

        degrees = np.array([G.degree[node] for node in candidates])
        P_degrees = degrees / degrees.sum()
        # prefer to attach to people in it's own cluster
        P_cluster = np.array([cluster_boost if clusters_ground_truth[node]
                              == cluster else 1 / cluster_boost for node in candidates])
        P = P_degrees * P_cluster

        while G.degree[c_node] < m:
            potential_node = np.random.randint(0, len(candidates))
            p = P[potential_node]
            if np.random.random() <= p:
                G.add_edge(c_node, potential_node)
            candidates = np.delete(candidates, potential_node)
            P = np.delete(P, potential_node)
            if len(candidates) <= 0:
                break

    print(len(G.edges), len(G.nodes))

    return G
