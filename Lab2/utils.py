import random

import numpy as np


def create_population(population_size=20, individual_size=10):  # TODO zmienić domyślne wartości
    population = []

    for i in range(population_size):
        individual = []
        for j in range(individual_size):
            individual.append(random.randint(0, 1))
        population.append(np.array(individual))

    return np.array(population)


def q(x):  # TODO a co w sytuacji, gdy prędkość będzie większa niż abs(v0) < 2 ?
    h0 = 200
    v0 = 0
    fuel = x.sum()
    m0 = 200 + fuel

    for i in x:
        m0 -= i * 1
        a = (45 / m0) * i - 0.09
        v0 = v0 + a
        h0 = h0 + v0 + a / 2

        if (0 <= h0 < 2) and abs(v0) < 2:
            return 2000 - fuel

        elif h0 < 0:
            return -1000 - fuel

    return - fuel


def score(q, P):
    return [q(x) for x in P]


def find_best(P, O):
    index = np.argmax(O)
    return P[index], O[index]


def selection(P, O, u):
    O = [o + (1000 + P.shape[1]) for o in O]
    O = [o / sum(O) for o in O]
    indices = np.arange(0, P.shape[0])
    selected_indices = np.random.choice(indices, u, replace=True, p=O)

    return P[selected_indices]