from typing import Dict, Any, Tuple, List

from src.algorithms.algorithm import Algorithm
from src.point import Point, generate_random_point


class RS(Algorithm):

    def run(self, n_rounds: int, verbose: bool = False) -> Tuple[Point, float, List]:
        for i in range(n_rounds):
            current_point: Point = self._random_point()
            quality: float = self._calculate_quality(current_point)
            if quality > self._best_quality:
                self._best_quality = quality
                self._best_point = current_point
            if verbose:
                self._log(i)
            self._quality_history.append(self._best_quality)
        return self._best_point, self._best_quality, self._quality_history

    def init_params(self, params: Dict[str, Any]) -> None:
        super().init_params(params)

    def _random_point(self) -> Point:
        return generate_random_point()
