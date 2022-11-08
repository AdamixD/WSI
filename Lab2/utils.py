import numpy as np


class GeneticAlgorithmAnalyzer:
    def __init__(self, solver, population_creator, problem, n_samples):
        self._solver = solver
        self._problem = problem
        self._population_creator = population_creator
        self._n_samples = n_samples
        self._X_best = []
        self._O_best = []

    def analyze(self):
        for i in range(self._n_samples):
            pop0 = self._population_creator.create_population()
            x_best, o_best = self._solver.solve(self._problem, pop0)
            self._X_best.append(x_best)
            self._O_best.append(o_best)

            print(f"Sample {i + 1}: Best score = {o_best}")

        o_best_total = np.max(self._O_best)
        o_min_total = np.min(self._O_best)
        o_best_total_index = np.argmax(self._O_best)
        average_score = np.average(self._O_best)
        median_score = np.median(self._O_best)

        print(' ')
        print(f'Best sample = {o_best_total_index + 1}')
        print(f'Best score = {o_best_total}, Minimum score = {o_min_total}')
        print(f'Average score = {average_score}, Median score = {median_score}')
