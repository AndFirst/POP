import sys
import os

# Pobierz aktualną ścieżkę do pliku experiments.py
current_path = os.path.dirname(os.path.abspath(__file__))

# Dodaj ścieżki do katalogów projektu
project_root = os.path.join(current_path, '..')
sys.path.append(project_root)

# Dodaj ścieżkę do katalogu src
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

# Dodaj ścieżki do katalogów experiments i plots
experiments_path = os.path.join(project_root, 'experiments')
sys.path.append(experiments_path)

plots_path = os.path.join(project_root, 'plots')
sys.path.append(plots_path)

from src.utils import eval_gini
import xgboost as xgb
from src.algorithms.algorithm import Algorithm
from src.algorithms.SA import SA
from src.algorithms.RS import RS
from src.algorithms.EA import EA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
import json
from collections import defaultdict


N_RUNS = 10
ALGORITHMS = RS, SA, EA
METRICS = 'gini', 'accuracy'
DATASETS = 'smoker_status', 'porto_seguro'

EA_EPOCHS = 50
RS_SA_EPOCHS = 300

CONST_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cpu'
}
NUM_BOOST_ROUND = 10

ALGORITHM_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cpu',

    "temperature": 100,
    "cooling": 0.95,
    "rounds_with_const_temperature": 10,

    "mutation_prob": 0.9,
    "mutation_rate": 0.1,
    "population_size": 10,
    "elite_size": 1,
}


def get_split_dataset(data: pd.DataFrame):
    Y = data['target'].values
    X = data.drop(columns='target').values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_algorithm(data, params, algorithm, metric,  n_rounds):
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_dataset(data)
    params.update(
        {"x_train": X_train,
         "x_test": X_val,
         "y_train": y_train,
         "y_test": y_val,
         "metric": metric})
    algorithm.init_params(params)
    point, quality, history = algorithm.run(n_rounds, verbose=False)

    xgb_params = CONST_XGB_PARAMS.copy()
    xgb_params.update(point.serialize())
    train_matrix = xgb.DMatrix(data=X_train, label=y_train)
    test_matrix = xgb.DMatrix(data=X_test)

    model = xgb.train(xgb_params, train_matrix,
                      num_boost_round=NUM_BOOST_ROUND)
    y_pred = model.predict(test_matrix)

    if metric == 'gini':
        score = eval_gini(y_test, y_pred)
    elif metric == 'accuracy':
        y_pred_labels = np.array([1 if pred > 0.5 else 0 for pred in y_pred])
        score = accuracy_score(y_test, y_pred_labels)
    else:
        raise ValueError('Wrong metric.')
    return point, quality, history, score


def main():
    best_results = defaultdict(list)
    all_iters = len(DATASETS) * len(METRICS) * len(ALGORITHMS) * N_RUNS
    current_iter = 0
    time_zero = time.time()
    for dataset_name in DATASETS:
        dataset = pd.read_csv(f"dataset/{dataset_name}.csv", index_col="id")
        for metric in METRICS:
            for algorithm_class in ALGORITHMS:
                algorithm_rounds = EA_EPOCHS if algorithm_class is EA else RS_SA_EPOCHS
                key = f'{algorithm_class.__name__}-{metric}-{dataset_name}'
                for i in range(N_RUNS):
                    os.system('clear')
                    print(key)
                    print(f'Progress: {current_iter}/{all_iters}')
                    print(f'Time elapsed: {(time.time() - time_zero) :.2f}')
                    algorithm = algorithm_class()
                    results = run_algorithm(
                        dataset, ALGORITHM_PARAMS, algorithm, metric, algorithm_rounds)
                    best_results[key].append(results)
                    current_iter += 1
    os.system('clear')
    print(f'Progress: {current_iter}/{all_iters}')
    print(f'Time elapsed: {(time.time() - time_zero) :.2f}')

    serialized = {}
    for k, v in best_results.items():
        serialized.update({k: []})
        for result in v:
            new_result = result[0].serialize(), result[1:]
            serialized[k].append(new_result)
    with open('results.json', 'w') as json_file:
        json.dump(serialized, json_file)


if __name__ == '__main__':
    main()
