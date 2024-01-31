from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from src.config import DEFAULT_PARAMS


class XGBoostParam(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self._name = config['name']

    @abstractmethod
    def serialize(self, numeric_value: float) -> Dict[str, str | float | int]:
        raise NotImplemented

    @abstractmethod
    def generate_random_param(self) -> float:
        raise NotImplemented

    @abstractmethod
    def get_mutated_point(self, numeric_value: float, sigma: float) -> float:
        raise NotImplemented
    
    def get_default(self)->float:
        return DEFAULT_PARAMS[self._name]


class ContinuousParam(XGBoostParam):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.min_value = config.get("min") or 0
        self.max_value = config.get("max") or 10

    def serialize(self, numeric_value: float) -> Dict[str, float]:
        return {self._name: numeric_value}

    def generate_random_param(self) -> float:
        return random.uniform(self.min_value, self.max_value)

    def get_mutated_point(self, numeric_value: float, sigma: float) -> float:
        mutated_value = numeric_value + sigma * random.gauss(0, 1)
        return np.clip(mutated_value, self.min_value, self.max_value)

class BinaryParam(XGBoostParam):
    def serialize(self, numeric_value: float) -> Dict[str, int]:
        return {self._name: int(numeric_value)}

    def generate_random_param(self) -> float:
        return random.randint(0, 1)

    def get_mutated_point(self, numeric_value: float, sigma: float) -> float:
        prob = 1 / (1 + np.exp(-sigma))
        return random.choices((numeric_value, 1 - numeric_value), [prob, 1 - prob])[0]


class CategoricalParam(XGBoostParam):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._categories = config.get("categories")

    def serialize(self, numeric_value: float) -> Dict[str, str]:
        reverse_categories_dict = {v: k for k, v in self._categories.items()}

        return {self._name: reverse_categories_dict.get(int(numeric_value))}

    def generate_random_param(self) -> float:
        return random.choice(list(self._categories.values()))

    def get_mutated_point(self, numeric_value: float, sigma: float) -> float:
        value = numeric_value + random.gauss(0, sigma)
        histogram, bin_edges = np.histogram([value], bins=len(self._categories), range=(0, len(self._categories) - 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return float(bin_centers[np.argmax(histogram)])


class DiscreteParam(XGBoostParam):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._min_value = config.get("min") or 0
        self._max_value = config.get("max") or 10

    def serialize(self, numeric_value: float) -> Dict[str, int]:
        return {self._name: int(numeric_value)}

    def generate_random_param(self) -> float:
        x = random.randint(self._min_value, self._max_value)
        return x

    def get_mutated_point(self, numeric_value: float, sigma: float) -> float:
        value = numeric_value + random.gauss(0, sigma)
        histogram, bin_edges = np.histogram([value], bins=self._max_value - self._min_value,
                                            range=(self._min_value, self._max_value))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return float(bin_centers[np.argmax(histogram)])


def params_factory(config: Dict[str, Any]) -> XGBoostParam:
    param_type = config.get("type")
    match param_type:
        case 'continuous':
            return ContinuousParam(config)
        case 'binary':
            return BinaryParam(config)
        case 'categorical':
            return CategoricalParam(config)
        case 'discrete':
            return DiscreteParam(config)
        case _:
            raise ValueError(f"Parameter type {param_type} not exist")
