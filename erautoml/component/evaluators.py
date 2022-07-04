import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
import ray
import time
import functools
import types


f1_macro = functools.partial(f1_score, average='macro')
f1_micro = functools.partial(f1_score, average='micro')
metric_dict_lb = {'f1': f1_score, 'f1_score': f1_score, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'roc_auc_score': roc_auc_score, \
    'accuracy': accuracy_score, 'acc': accuracy_score, 'accuracy_score': accuracy_score}
metric_dict_sb = {'mae': mean_absolute_error}

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)


class Metric:

    def __init__(self, func=None, larger_better=True):

        if isinstance(func, str) and (func in metric_dict_lb):
            self.func = metric_dict_lb[func]
            self.larger_better = True
        elif isinstance(func, str) and (func in metric_dict_sb):
            self.func = metric_dict_sb[func]
            self.larger_better = False
        elif func:
            self.func = func
            self.larger_better = larger_better
        else:
            self.func = None

        return

    def compare(self, y_true, y_pred):
        if not self.func:
            print('ERROR: function of metric not initialized!')
            return
        return self.func(y_true, y_pred)


@ray.remote
def evaluate(algo_ins, hps, dataset, metric, k_fold=5):
    start = time.time()
    model = algo_ins
    model.generate_model(hps)
    eval_values = []
    data_split = KFold(n_splits=k_fold, shuffle=False).split(dataset.features, dataset.labels)
    for train_index, valid_index in data_split:
        train_features, train_labels = dataset.features[train_index], dataset.labels[train_index]
        valid_features, valid_labels = dataset.features[valid_index], dataset.labels[valid_index]
        try:
            model.fit(train_features, train_labels)
        except ValueError as e:
            print(algo_ins.__class__.__name__, "model fit error: {}".format(e))
            # if metric.larger_better:
            #     return {'arm': algo_ins.__class__.__name__, 'hps': hps, 'score': -9e16, 'eval_time': -1}
            # else:
            #     return {'arm': algo_ins.__class__.__name__, 'hps': hps, 'score': 9e16, 'eval_time': -1}

        predictions = model.predict(valid_features)
        this_value = metric.compare(valid_labels, predictions)
        eval_values.append(this_value)

    score = np.mean(np.array(eval_values))

    eval_time = time.time() - start

    return {'arm': algo_ins.__class__.__name__, 'hps': hps, 'score': score, 'eval_time': eval_time}


def evaluate_serial(algo_ins, hps, dataset, metric, k_fold=5):
    start = time.time()
    model = algo_ins
    model.generate_model(hps)
    eval_values = []
    data_split = KFold(n_splits=k_fold, shuffle=False).split(dataset.features, dataset.labels)
    for train_index, valid_index in data_split:
        train_features, train_labels = dataset.features[train_index], dataset.labels[train_index]
        valid_features, valid_labels = dataset.features[valid_index], dataset.labels[valid_index]
        try:
            model.fit(train_features, train_labels)
        except ValueError as e:
            print(algo_ins.__class__.__name__, "model fit error: {}".format(e))
            # if metric.larger_better:
            #     return {'arm': algo_ins.__class__.__name__, 'hps': hps, 'score': -9e16, 'eval_time': -1}
            # else:
            #     return {'arm': algo_ins.__class__.__name__, 'hps': hps, 'score': 9e16, 'eval_time': -1}

        predictions = model.predict(valid_features)
        this_value = metric.compare(valid_labels, predictions)
        eval_values.append(this_value)

    score = np.mean(np.array(eval_values))

    eval_time = time.time() - start

    return {'arm': algo_ins.__class__.__name__, 'hps': hps, 'score': score, 'eval_time': eval_time}


if __name__ == "__main__":
    print("hello world!")
