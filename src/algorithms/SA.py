import random
from typing import Dict, Any, Tuple, List

import numpy as np

from src.algorithms.algorithm import Algorithm
from src.point import Point, generate_random_point


class SA(Algorithm):
    def __init__(self):
        super().__init__()
        self._temperature = None
        self._cooling = None
        self._rounds_with_const_temperature = None

        self._work_point = None
        self._work_point_quality = None

    def run(self, n_rounds: int, verbose: bool = False) -> Tuple[Point, float, List]:
        self._work_point = generate_random_point()
        self._work_point_quality = self._calculate_quality(self._work_point)
        self._best_point = self._work_point
        self._best_quality = self._work_point_quality

        for i in range(n_rounds):
            new_point = self._work_point.neighbour()
            new_point_quality = self._calculate_quality(new_point)

            if new_point_quality > self._work_point_quality:
                self._work_point = new_point
                self._work_point_quality = new_point_quality
            else:
                delta = self._work_point_quality - new_point_quality
                if random.uniform(0, 1) < np.exp(-delta / self._temperature):
                    self._work_point_quality = new_point_quality
                    self._work_point = new_point

            if new_point_quality > self._best_quality:
                self._best_point = new_point
                self._best_quality = new_point_quality

            if i % self._rounds_with_const_temperature == 0:
                self._temperature *= self._cooling
            if verbose:
                self._log(i)
            self._quality_history.append(self._best_quality)
        return self._best_point, self._best_quality, self._quality_history

    def init_params(self, params: Dict[str, Any]) -> None:
        super().init_params(params)
        self._temperature = params['temperature']
        self._cooling = params['cooling']
        self._rounds_with_const_temperature = params['rounds_with_const_temperature']
