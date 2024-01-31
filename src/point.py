from __future__ import annotations
import random
import copy
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.params import XGBoostParam, params_factory
from src.utils import eval_gini
from src.config import BOOSTERS, TREE_PARAMS


class Booster:
    def __init__(self, name: str, params: List[XGBoostParam]) -> None:
        self._name: str = name
        self._params: List[XGBoostParam] = params

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if value not in BOOSTERS:
            raise ValueError('Invalid booster')
        self._name = value

    @property
    def length(self):
        return len(self._params)

    @property
    def params(self):
        return self._params


class Point:
    def __init__(self, booster: Booster) -> None:
        self._booster: Booster = booster
        self._current_booster: Booster = booster
        self._params: List[float] = list()
        self._init_numeric_params()

    @property
    def params(self) -> List[float]:
        return self._params

    @params.setter
    def params(self, value: List[float]) -> None:
        self._params = value

    @property
    def current_booster(self) -> Booster:
        return self._current_booster

    def swap_current_booster(self) -> None:
        pass
        # self._current_booster = self._boosters[0] if self._current_booster == self._boosters[-1] else self._boosters[-1]

    def neighbour(self) -> Point:
        new_point = copy.deepcopy(self)
        first_index = new_point._get_first_param_index()
        param_to_swap = random.choice(new_point._current_booster.params)
        for i, param in enumerate(new_point._current_booster.params):
            if param == param_to_swap:
                new_value = param.generate_random_param()
                new_point._params[first_index + i] = new_value
        return new_point


    def mutate(self, mutation_prob: float, mutation_rate: float) -> Point:
        new_point = copy.deepcopy(self)
        first_index = new_point._get_first_param_index()
        for i, param in enumerate(new_point._current_booster.params):
            if random.uniform(0, 1) < mutation_prob:
                new_value = param.get_mutated_point(new_point.params[first_index + i], mutation_rate)
                new_point._params[first_index + i] = new_value
        return new_point

    def _get_first_param_index(self) -> int:
        return 0

    def _params_slice(self) -> List[float]:
        return self._params
    
    def serialize(self) -> Dict[str, str | float | int]:
        booster_params: Dict[str, str | float | int] = {"booster": self._current_booster.name}
        for numeric_param, abstract_param in zip(self._params, self._current_booster.params):
            booster_params.update(abstract_param.serialize(numeric_param))
        return booster_params

    def _init_numeric_params(self) -> None:
        params = []
        for param in self._booster.params:
            params.append(param.get_default())
            # params.append(param.generate_random_param())
        self._params = params


def generate_random_point() -> Point:
    tree_params = [params_factory(config) for config in TREE_PARAMS]
    tree_booster = Booster('gbtree', tree_params)
    
    point = Point(tree_booster)
    return point
