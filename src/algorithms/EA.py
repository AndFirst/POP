import random
from copy import deepcopy
from typing import Dict, Any, Tuple, List

from numpy import argmax

from src.algorithms.algorithm import Algorithm
from src.point import Point, generate_random_point


class EA(Algorithm):
    def __init__(self):
        super().__init__()

        self._population_size = None
        self._mutation_prob = None
        self._mutation_rate = None
        self._elite_size = None

        self._population = None
        self._population_quality = None

        self._best_mutant = None
        self._best_mutant_quality = None

        self._current_mutants = None
        self._current_mutants_quality = None

    def run(self, n_rounds: int, verbose: bool = False) -> Tuple[Point, float, List]:
        self._init_population()
        self._population_quality = self._calculate_group_quality(self._population)
        self._best_point, self._best_quality = self._find_best(self._population, self._population_quality)
        for i in range(n_rounds):
            self._reproduction()
            self._mutation()

            self._current_mutants_quality = self._calculate_group_quality(self._current_mutants)
            self._best_mutant, self._best_mutant_quality = self._find_best(self._current_mutants,
                                                                           self._current_mutants_quality)

            if self._best_mutant_quality > self._best_quality:
                self._best_quality = self._best_mutant_quality
                self._best_point = self._best_mutant

            self._succession()
            self._current_mutants = None
            self._best_mutant = None
            if verbose:
                self._log(i)
            self._quality_history.append(self._best_quality)
        return self._best_point, self._best_quality, self._quality_history

    def init_params(self, params: Dict[str, Any]) -> None:
        super().init_params(params)
        self._mutation_rate = params.get("mutation_rate")
        self._mutation_prob = params.get("mutation_prob")
        self._elite_size = params.get("elite_size")
        self._population_size = params.get("population_size")

    def _init_population(self):
        self._population = [generate_random_point() for _ in range(self._population_size)]

    def _calculate_group_quality(self, population):
        return [self._calculate_quality(point) for point in population]

    def _find_best(self, population, population_quality):
        best_quality = max(population_quality)
        best_index = argmax(population_quality)
        best_point = population[best_index]
        return best_point, best_quality

    def _reproduction(self):
        new_population = []
        for _ in range(self._population_size):
            first_index = random.randint(0, self._population_size - 1)
            second_index = random.randint(0, self._population_size - 1)
            winner_index = first_index if self._population_quality[first_index] >= self._population_quality[
                second_index] else second_index
            new_population.append(deepcopy(self._population[winner_index]))
        self._current_mutants = new_population

    def _mutation(self):
        new_population = []
        for point in self._current_mutants:
            new_population.append(point.mutate(self._mutation_prob, self._mutation_rate))
        self._current_mutants = new_population

    def _succession(self):
        if self._elite_size == 0:
            self._population = self._current_mutants
        self._population, self._population_quality = self._sort(self._population, self._population_quality)
        self._current_mutants, self._current_mutants_quality = self._sort(self._current_mutants,
                                                                          self._current_mutants_quality)

        new_population = self._population[:self._elite_size] + self._current_mutants[:-self._elite_size]
        new_population_quality = self._population_quality[:self._elite_size] + self._current_mutants_quality[
                                                                               :-self._elite_size]
        self._population = new_population
        self._population_quality = new_population_quality

    def _sort(self, population, population_quality) -> Tuple:
        zipped_lists = list(sorted(zip(population_quality, population), key=lambda x: x[0], reverse=True))
        sorted_quality, sorted_population = [a for a, b in zipped_lists], [b for a, b in zipped_lists]
        return sorted_population, sorted_quality
