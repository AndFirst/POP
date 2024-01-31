BOOSTERS = ("gbtree", "gblinear")

TREE_PARAMS = [
    {"name": "learning_rate", "type": "continuous", "min": 0, "max": 1},
    {"name": "min_split_loss", "type": "continuous", "min": 0, "max": 100},
    {"name": "max_depth", "type": "discrete", "min": 1, "max": 20},
    {"name": "min_child_weight", "type": "continuous", "min": 0},
    {"name": "max_delta_step", "type": "continuous", "min": 0},
    {"name": "subsample", "type": "continuous", "min": 0, "max": 1},
    # {"name": "colsample_bytree", "type": "continuous", "min": 0, "max": 1},
    # {"name": "colsample_bylevel", "type": "continuous", "min": 0, "max": 1},
    # {"name": "colsample_bynode", "type": "continuous", "min": 0, "max": 1},
    {"name": "reg_lambda", "type": "continuous", "min": 0},
    {"name": "reg_alpha", "type": "continuous", "min": 0},
    # {"name": "tree_method", "type": "categorical", "categories": {"approx": 0, "hist": 1}},
    {"name": "refresh_leaf", "type": "binary"},
    {"name": "grow_policy", "type": "categorical",
        "categories": {"depthwise": 0, "lossguide": 1}},
    {"name": "max_leaves", "type": "discrete", "min": 0},
    {"name": "max_bin", "type": "discrete", "min": 2, "max": 300},
    {"name": "num_parallel_tree", "type": "discrete", "min": 1}
]

# LINEAR_PARAMS = [
#     {"name": "reg_lambda", "type": "continuous", "min": 0, "max": 100},
#     {"name": "reg_alpha", "type": "continuous", "min": 0, "max": 100},
#     {"name": "feature_selector", "type": "categorical",
#      "categories": {"cyclic": 0, "shuffle": 1}},
# ]

DEFAULT_PARAMS = {
    "learning_rate": 0.5394889061658026,
    "min_split_loss": 11.45391288874038,
    "max_depth": 15,
    "min_child_weight": 8.451923662989186,
    "max_delta_step": 5.292893287078815,
    "subsample": 0.7009570493769901,
    "reg_lambda": 1.3098532210002334,
    "reg_alpha": 0.21972098319276268,
    "refresh_leaf": 1,
    "grow_policy": 1,
    "max_leaves": 0,
    "max_bin": 290,
    "num_parallel_tree": 10
}

# DEFAULT_PARAMS = {
#     "learning_rate": 0.44672685772902465,
# "min_split_loss": 3.151639067699963,
# "max_depth": 20,
# "min_child_weight": 4.270318315887933,
# "max_delta_step": 3.8318224172763227,
# "subsample": 0.8150585506365398,
# "reg_lambda": 1.3086560582963391,
# "reg_alpha": 5.491929660584046,
# "refresh_leaf": 0,
# "grow_policy": 1,
# "max_leaves": 0,
# "max_bin": 277,
# "num_parallel_tree": 8

# }
