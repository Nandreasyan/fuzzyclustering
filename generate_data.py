import numpy as np
import networkx as nx


def skills_gen(skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits):
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
        user_skills = 1*user_skills
        nb_skill_sets = np.random.randint(min_skill_sets, max_skill_sets)
        skills_sets_indices = np.random.choice(
            range(len(skills_sets)), nb_skill_sets)

        for s in skills_sets_indices:
            skill_set = skills_sets[s]
            for skill in skill_set:
                i = all_skills.index(skill)
                user_skills[i] = 1

        nb_edits = np.random.randint(min_edits, max_edits)
        for _ in range(nb_edits):
            # flip a random bit
            nbTrue = user_skills.sum()
            nbFalse = len(user_skills) - nbTrue

            a = np.zeros((len(user_skills)))
            nonzero_idxs = np.where(user_skills != 0)[0]
            a[nonzero_idxs] = nbTrue
            a[np.logical_not(user_skills)] = nbFalse
            #p = np.array(user_skills)
            #p /= p.sum()
            p = np.full(len(user_skills), 0.5) / a

            i = np.random.choice(range(len(all_skills)), p=p)
            user_skills[i] ^= 1

        users_skills.append(user_skills)
        clusters_ground_truth.append(skills_sets_indices[0])

    users_skills = np.array(users_skills)
    clusters_ground_truth = np.array(clusters_ground_truth)

    return users_skills, clusters_ground_truth


def skills_gen_fz(skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits):
    """Generate N random users_skills_fz levels based on skills_sets
    add a bit of noise with the min and max edits
    """
    all_skills = list()
    for ss in skills_sets:
        all_skills += ss

    users_skills_fz = []
    clusters_ground_truth = []

    # knowledge level of skills for every user
    skills_lv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for _ in range(N):
        user_skills = np.zeros((len(all_skills)), dtype=float)

        nb_skill_sets = np.random.randint(min_skill_sets, max_skill_sets)
        skills_sets_indices = np.random.choice(
            range(len(skills_sets)), nb_skill_sets)

        for s in skills_sets_indices:
            skill_set = skills_sets[s]
            for skill in skill_set:
                i = all_skills.index(skill)
                user_skills[i] = np.random.choice(skills_lv)

        nb_edits = np.random.randint(min_edits, max_edits)
        for _ in range(nb_edits):
            # flip a random bit
            nbTrue = user_skills.sum()
            nbFalse = len(user_skills) - nbTrue

            a = np.zeros((len(user_skills)))
            nonzero_idxs = np.where(user_skills != 0)[0]
            a[nonzero_idxs] = nbTrue
            a[np.logical_not(user_skills)] = nbFalse
            p = np.array(user_skills)
            p /= p.sum()  # normalize
            #p = np.full((len(user_skills)), 0.5) / a

            i = np.random.choice(range(len(all_skills)), p=p)
            user_skills[i] = np.random.choice(skills_lv)

        users_skills_fz.append(user_skills)
        clusters_ground_truth.append(skills_sets_indices[0])

    users_skills_fz = np.array(users_skills_fz)
    clusters_ground_truth = np.array(clusters_ground_truth)

    return users_skills_fz, clusters_ground_truth

def testfz(N):
    X = np.concatenate((
        np.random.normal((-2, -2), size=(N, 2)),
        np.random.normal((2, 2), size=(N, 2)),
        np.random.normal((9, 0), size=(N, 2)),
        np.random.normal((5, -8), size=(N, 2))
    ))

    return X

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
