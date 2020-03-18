import numpy as np

from .model import generate_lgbbo
from .model import generate_models
from .swap import get_swap_table
from .swap import swap_x_data
from .swap import swap_y_data


def train_models(x_train, y_train):
    swap_table = get_swap_table(x_train)
    x_train = swap_x_data(x_train, swap_table)
    y_train = swap_y_data(y_train, swap_table)

    lgb_bo = generate_lgbbo(x_train, y_train)
    models = generate_models(lgb_bo, x_train, y_train)

    return models, lgb_bo, swap_table

def test_models(models, swap_table, x_test):
    x_test = swap_x_data(x_test, swap_table)

    preds = []
    for model in models:
        pred = model.predict_proba(x_test)[:, 1]
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    pred = swap_y_data(pred, swap_table)

    return pred
