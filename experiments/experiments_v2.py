from collections import defaultdict
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.algorithms.EA import EA
from src.algorithms.RS import RS
from src.algorithms.SA import SA
from src.algorithms.algorithm import Algorithm
import xgboost as xgb
import os
from src.utils import eval_gini


# N_RUNS = 10
# ALGORITHMS = RS, SA, EA
# METRICS = 'gini', 'accuracy'
# DATASETS = 'smoker_status', 'porto_seguro'

N_RUNS = 10
ALGORITHMS = SA,
METRICS = 'gini',
DATASETS = 'porto_seguro',

EA_EPOCHS = 100
RS_SA_EPOCHS = 1000

CONST_XGB_PARAMS = {        
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda'
}
NUM_BOOST_ROUND = 10

ALGORITHM_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'device': 'cuda',
    
    "temperature": 100,
    "cooling": 0.95,
    "rounds_with_const_temperature": 10,
    
    "mutation_prob": 0.5,
    "mutation_rate": 0.05,
    "population_size": 20,
    "elite_size": 1,
}
    



def get_split_dataset(data: pd.DataFrame):
    Y = data['target'].values
    X = data.drop(columns='target').values
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
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

    model = xgb.train(xgb_params, train_matrix, num_boost_round=NUM_BOOST_ROUND)
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
                    results = run_algorithm(dataset, ALGORITHM_PARAMS, algorithm, metric, algorithm_rounds)
                    best_results[key].append(results)
                    current_iter += 1
    os.system('clear')
    print(f'Progress: {current_iter}/{all_iters}')
    print(f'Time elapsed: {(time.time() - time_zero) :.2f}')

    serialized = {}
    for k, v in best_results.items():
        serialized.update({k:[]})
        for result in v:
            new_result = result[0].serialize(), result[1:]
            serialized[k].append(new_result)
    with open('results_12.json', 'w') as json_file:
        json.dump(serialized, json_file)

if __name__ == '__main__':
    main()    




