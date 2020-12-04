import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import collections
import random


def generate_skills(skills_sets, N, min_skill_sets, max_skill_sets, min_edits, max_edits):
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

    return users_skills, clusters_ground_truth


def generate_graph(N, K, P, seed):
    G = nx.watts_strogatz_graph(N, K, P, seed)
    return G
