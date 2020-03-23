import json

import numpy as np
import pandas as pd

from .model import create_forced_splits_tree
from .model import generate_lgbbo
from .model import generate_models
from .swap import get_swap_table
from .swap import swap_x_data
from .swap import swap_y_data


def train_models(x_train, y_train):
    swap_table = get_swap_table(x_train)
    x_train = swap_x_data(x_train, swap_table)
    y_train = swap_y_data(y_train, swap_table)

    splits_filename = 'forced_splits_0.json'
    with open(splits_filename, 'w') as f:
        splits_tree = create_forced_splits_tree(x_train)
        json.dump(splits_tree, f)

    lgb_bo = generate_lgbbo(x_train, y_train, forced_splits=splits_filename)
    models = generate_models(lgb_bo, x_train, y_train, forced_splits=splits_filename)

    return models, lgb_bo

def test_models(models, x_test):
    swap_table = get_swap_table(x_test)
    x_test = swap_x_data(x_test, swap_table)

    preds = []
    for model in models:
        pred = model.predict_proba(x_test)[:, 1]
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    pred = swap_y_data(pred, swap_table)

    submission = pd.DataFrame(data=pred, index=x_test.index, columns=['winner'])
    submission.index.name = 'game_id'

    return submission
