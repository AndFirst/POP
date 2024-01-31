import json
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

from src.point import Point
from src.utils import eval_gini


class Algorithm(ABC):
    def __init__(self):
        self._current_booster = None
        self._y_test = None
        self._train_matrix = None
        self._test_matrix = None
        self._num_boost_rounds = 10
        self._output_file = None

        self._metric = "accuracy"
        self._objective = "binary:logistic"
        self._eval_metric = "logloss"
        self._device = "cuda"

        self._best_point = None
        self._best_quality = 0
        self._quality_history = []

    @abstractmethod
    def run(self, n_rounds: int, verbose: bool = False) -> Tuple[Point, float, List]:
        raise NotImplemented

    @abstractmethod
    def init_params(self, params: Dict[str, Any]) -> None:
        self._train_matrix = xgb.DMatrix(
            data=params["x_train"], label=params["y_train"])
        self._test_matrix = xgb.DMatrix(data=params["x_test"])
        self._y_test = params.get('y_test')

        if params.get("output_file"):
            self._output_file = params.get("output_file")

        if params.get("num_boost_rounds"):
            self._num_boost_rounds = params.get("num_boost_rounds")

        if params.get("metric"):
            self._metric = params.get("metric")

        if params.get("objective"):
            self._objective = params.get('objective')

        if params.get("eval_metric"):
            self._eval_metric = params.get('eval_metric')

        if params.get("device"):
            self._device = params.get('device')

    def _log(self, i: int) -> None:
        if self._output_file is None:
            print(f"\n"
                  f"==============================\n"
                  f"Round: {i}\n"
                  f"Best quality: {self._best_quality:.4f}\n"
                  f"Best point: {self._best_point.serialize()}\n"
                  f"==============================\n")
        else:
            with open(self._output_file, "a") as file:
                output = {"Round": i,
                          "Best quality": round(self._best_quality, 4),
                          "Best point": self._best_point.serialize()}
                json.dump(output, file)

    def _calculate_quality(self, point: Point) -> float:
        params = point.serialize()
        params.update({'objective': self._objective,
                      'eval_metric': self._eval_metric, 'device': self._device})
        model = xgb.train(params, self._train_matrix, self._num_boost_rounds)
        y_pred = model.predict(self._test_matrix)

        if self._metric == 'gini':
            score = eval_gini(self._y_test, y_pred)
        else:
            y_pred_labels = np.array(
                [1 if pred > 0.5 else 0 for pred in y_pred])
            score = accuracy_score(self._y_test, y_pred_labels)
        return score
