import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from functools import partial
import lightgbm as lgb


def create_forced_splits_tree(x_data):
    p0_species = x_data.columns.get_loc('p0_species')
    p1_species = x_data.columns.get_loc('p1_species')

    return {
        "feature": p0_species,
        "threshold": 0,
        "left": {
            "feature": p1_species,
            "threshold": 0,
            "right": {
                "feature": p1_species,
                "threshold": 1,
            },
        },
        "right": {
            "feature": p0_species,
            "threshold": 1,
            "left": {
                "feature": p1_species,
                "threshold": 1,
            },
        }
    }


def lgb_cv(num_leaves, learning_rate, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda, bagging_fraction, feature_fraction, x_data=None, y_data=None, n_splits=5, forced_splits=None, output='score'):
    score = 0
    kf = KFold(n_splits=n_splits)
    models = []
    for train_index, valid_index in kf.split(x_data):
        x_train, y_train = x_data.iloc[train_index], y_data[train_index]
        x_valid, y_valid = x_data.iloc[valid_index], y_data[valid_index]
        
        model = lgb.LGBMClassifier(
            num_leaves = int(num_leaves), 
            learning_rate = learning_rate, 
            n_estimators = int(n_estimators), 
            subsample = np.clip(subsample, 0, 1), 
            colsample_bytree = np.clip(colsample_bytree, 0, 1), 
            reg_alpha = reg_alpha, 
            reg_lambda = reg_lambda,
            bagging_fraction = bagging_fraction,
            feature_fraction = feature_fraction,
            forced_splits = forced_splits,
        )
        
        model.fit(x_train, y_train)
        models.append(model)
        
        pred = model.predict_proba(x_valid)[:, 1]
        true = y_valid
        score += roc_auc_score(true, pred)/n_splits
    
    if output == 'score':
        return score
    if output == 'model':
        return models


def generate_lgbbo(x_train, y_train, forced_splits=None):
    # 모델과 관련없는 변수 고정
    func_fixed = partial(lgb_cv, forced_splits=forced_splits, x_data=x_train, y_data=y_train, n_splits=5, output='score') 
    # 베이지안 최적화 범위 설정
    lgbBO = BayesianOptimization(
        func_fixed, 
        {
            'num_leaves': (256, 2048),        # num_leaves,       범위(16~1024)
            'learning_rate': (0.0001, 0.1),  # learning_rate,    범위(0.0001~0.1)
            'n_estimators': (16, 1024),      # n_estimators,     범위(16~1024)
            'subsample': (0, 0.5),             # subsample,        범위(0~1)
            'colsample_bytree': (0, 1),      # colsample_bytree, 범위(0~1)
            'reg_alpha': (0, 10),            # reg_alpha,        범위(0~10)
            'reg_lambda': (0, 50),           # reg_lambda,       범위(0~50)
            'bagging_fraction': (0.5, 1.0),
            'feature_fraction': (0.5, 1.0),
        }, 
        random_state=4321                    # 시드 고정
    )
    lgbBO.maximize(init_points=5, n_iter=5) # 처음 5회 랜덤 값으로 score 계산 후 30회 최적화

    # 이 예제에서는 7개 하이퍼 파라미터에 대해 30회 조정을 시도했습니다.
    # 다양한 하이퍼 파라미터, 더 많은 iteration을 시도하여 최상의 모델을 얻어보세요!
    # LightGBM Classifier: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

    return lgbBO


def generate_models(lgb_bo, x_train, y_train, forced_splits=None):
    params = lgb_bo.max['params']
    models = lgb_cv(
        params['num_leaves'], 
        params['learning_rate'], 
        params['n_estimators'], 
        params['subsample'], 
        params['colsample_bytree'], 
        params['reg_alpha'], 
        params['reg_lambda'],
        params['bagging_fraction'],
        params['feature_fraction'],
        forced_splits=forced_splits,
        x_data=x_train, y_data=y_train, n_splits=5, output='model')
    return models
