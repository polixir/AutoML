import configparser
import copy
import random
import sys
import time

import numpy as np
import psutil
from sklearn.metrics import f1_score

from .component.arms import Arms
from .algorithm.clfs import (GBDT, LDA, QDA, SGD, SVC, AdaBoost,
                                  BernoulliNB, DecisionTree, ExtraTrees,
                                  GaussianNB, HyperParameter, KNeighbors,
                                  LightGBM, LinearSVC, MultinomialNB,
                                  PassiveAggressive, RandomForest)
from .component.datasets import AutomlDataset
from .component.evaluators import Metric
from .algorithm.rgrs import (SVR, DecisionTreeRGR, LightGBMRGR, KNNRGR, 
                                RandomForestRGR, AdaBoostRGR, ExtraTreesRGR,
                                GBRT)
from .index import all_clf_class, all_rgr_class
from .component.strategies import ENUCBStrategy, ERUCBStrategy, run_strategy_parallel, RandomStrategy


class MyConfigParser(configparser.ConfigParser):
    def __init__(self, defaults=None):
        configparser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    return False


def set_hp_space_from_ini(cf, algo):
    if algo.__name__ in cf.sections():
        hp_space = []
        specified_params = {}
        for param_name, param_detail in cf.items(algo.__name__):
            param_detail_lst = param_detail.split(':')
            # user specified a param
            if len(param_detail_lst) == 1:
                if is_number(param_detail):
                    specified_params[param_name] = eval(param_detail)
                else:
                    specified_params[param_name] = param_detail
            # user specified an hp_space
            elif len(param_detail_lst) == 2:
                param_type, param_space = param_detail.split(':')
                if param_type == 'int':
                    hp_space.append(HyperParameter.int_hp(
                        param_name, eval(param_space)))
                elif param_type == 'cat':
                    hp_space.append(HyperParameter.categorical_hp(
                        param_name, eval(param_space)))
                elif param_type == 'float':
                    hp_space.append(HyperParameter.float_hp(
                        param_name, eval(param_space)))
            else:
                print('Error, too many colons!')
        return algo(hp_space=hp_space, specified_params=specified_params)
    else:
        return algo()


class BaseAutoml(object):

    def __init__(self):
        self.history_by_order = []
        self.history_by_score = []
        self.ensemble_info = []
        self.ensemble_models = []
        self.ensemble_weights = []
        self.arms = []
        self.strategy = None
        self.metric = None
        self.task = None
        return

    def dump_ini(self, filename=None):
        cf = MyConfigParser()
        if filename is None:
            if self.task == 'clf':
                index = all_clf_class
                filename = 'clf_algo.ini'
            elif self.task == 'rgr':
                index = all_rgr_class
                filename = 'rgr_algo.ini'
        for algo_class in index:
            class_name = algo_class.__name__
            cf.add_section(class_name)
            algo_ins = algo_class()
            for space in algo_ins.hp_space:
                if space.is_float_type():
                    content = 'float:' + str(space.hp_range)
                    cf.set(class_name, space.get_hp_name(), content)
                elif space.is_int_type():
                    content = 'int:' + str(space.hp_range)
                    cf.set(class_name, space.get_hp_name(), content)
                elif space.is_categorical_type():
                    content = 'cat:' + str(space.hp_range)
                    cf.set(class_name, space.get_hp_name(), content)
            if len(algo_ins.specified_params) > 0:
                for param_name in algo_ins.specified_params:
                    param_value = algo_ins.specified_params[param_name]
                    cf.set(class_name, param_name, str(param_value))
        with open(filename, "w+") as f:
            cf.write(f)

    def fit(self, X, y, max_run_time=None, total_pulls=None, ensemble_strategy=[4, 2], metric=None, parallel_num=-1, stationary=True, include_algo=None, algo_config_file=None):

        dataset = AutomlDataset(X, y)
        if self.task == 'clf':
            self.num_class = len(np.unique(y))
            if metric is None:
                metric = 'f1_macro'
            if not include_algo:
                include_algo = [AdaBoost, ExtraTrees, GaussianNB, GBDT, KNeighbors, LDA,
                                LinearSVC, PassiveAggressive, RandomForest, SVC, LightGBM, SGD]
                if X.shape[0] > 20000:
                    include_algo = [x for x in include_algo if x not in [SVC]]
            if algo_config_file is None:
                algo_config_file = 'clf_algo.ini'

        elif self.task == 'rgr':
            if metric is None:
                metric = 'mae'
            if not include_algo:
                include_algo = [DecisionTreeRGR, SVR, LightGBMRGR, KNNRGR, 
                                RandomForestRGR, AdaBoostRGR, ExtraTreesRGR, GBRT]
            if algo_config_file is None:
                algo_config_file = 'rgr_algo.ini'

        if max_run_time is not None:
            time_to_stop = time.time() + max_run_time
        else:
            if total_pulls is None:
                print('ERROR: Both [max_run_time] and [total_pulls] not set!')
                exit()
            time_to_stop = float('inf')
        if total_pulls is None:
            total_pulls = sys.maxsize

        X = np.array(X)
        y = np.array(y)

        if parallel_num == -1:
            parallel_num = psutil.cpu_count()

        if isinstance(metric, Metric):
            self.metric = metric
        elif isinstance(metric, str):
            self.metric = Metric(metric)

        cf = MyConfigParser()
        if len(cf.read(algo_config_file)) > 0:
            print('Using classifier algorithm config file:', algo_config_file)
            exist_config_file = True
        else:
            exist_config_file = False

        for algo in include_algo:
            if exist_config_file:
                algo_ins = set_hp_space_from_ini(cf, algo)
            else:
                algo_ins = algo()
            arm = Arms(dataset=dataset, algo_ins=algo_ins, metric=self.metric,
                       time_to_stop=time_to_stop, num_tasks=parallel_num)
            self.arms.append(arm)

        theta = 0.1
        beta = 0.5
        gamma = 1.0

        if stationary:
            self.strategy = ERUCBStrategy(arms=self.arms, theta=theta, beta=beta, gamma=gamma)
        else:
            self.strategy = ENUCBStrategy(arms=self.arms, t_axis_scale=0.025,
                                      alpha=4.0, theta=0.01, gamma=gamma, init_size=2, g_name='sigmoid')
        self.history_by_order = run_strategy_parallel(
            strategy=self.strategy, total_pulls=total_pulls, time_to_stop=time_to_stop, parallel_num=parallel_num, task=self.task)
        self.history_by_score = sorted(self.history_by_order, key=lambda X: X['score'], reverse=True) if self.metric.larger_better \
            else sorted(self.history_by_order, key=lambda X: X['score'], reverse=False)

        total_tn = 0

        if isinstance(ensemble_strategy, list):

            selected_algos = []
            for this_history in self.history_by_score:
                arm_name = this_history['arm']
                if arm_name not in selected_algos:
                    selected_algos.append(arm_name)
                if len(selected_algos) >= len(ensemble_strategy):
                    break

            expected_ensemble_num = dict(
                zip(selected_algos, ensemble_strategy))
            now_ensemble_num = dict(
                zip(selected_algos, [0 for _ in range(len(ensemble_strategy))]))

            for this_history in self.history_by_score:

                arm_name = this_history['arm']

                if (arm_name in selected_algos) and (now_ensemble_num[arm_name] < expected_ensemble_num[arm_name]):
                    self.ensemble_info.append(this_history)
                    hp = this_history['hps']
                    arm = self.strategy.ii[arm_name]
                    model = arm.algo_ins
                    model.generate_model(hp)
                    model.fit(X, y)
                    self.ensemble_models.append(copy.deepcopy(model))
                    self.ensemble_weights.append(self.strategy.tn[arm_name])
                    total_tn += self.strategy.tn[arm_name]
                    now_ensemble_num[arm_name] += 1

                if len(self.ensemble_models) == np.sum(ensemble_strategy):
                    break

        elif isinstance(ensemble_strategy, int):

            for i in range(ensemble_strategy):

                if i >= len(self.history_by_score):
                    break

                this_history = self.history_by_score[i]
                arm_name = this_history['arm']
                self.ensemble_info.append(this_history)
                hp = this_history['hps']
                arm = self.strategy.ii[arm_name]
                model = arm.algo_ins
                model.generate_model(hp)
                model.fit(X, y)
                self.ensemble_models.append(copy.deepcopy(model))
                self.ensemble_weights.append(self.strategy.tn[arm_name])
                total_tn += self.strategy.tn[arm_name]

        self.ensemble_weights = [tn / total_tn for tn in self.ensemble_weights]

        return

    def predict(self, X):
        return

    def predict_separate(self, X, y):

        ret = []

        if len(self.ensemble_models) != len(self.ensemble_weights):
            print('ERROR: different numbers between models and weighs')
            return

        for idx, this_history in enumerate(self.ensemble_info):
            model = self.ensemble_models[idx]
            score = self.metric.func(y, model.predict(X))
            ret.append({'model': this_history['arm'], 'weight': self.ensemble_weights[idx],
                       'cv': this_history['score'], 'test': score, 'hps': model.get_param_dict(this_history['hps'])})

        return ret

    def show_models(self, print_res=True):
        result_str = '=================ensemble models=================\n'
        for idx, this_history in enumerate(self.ensemble_info):
            name = this_history['arm']
            result_str += f"[{name}]\n"
            result_str += f"[weight: {str(self.ensemble_weights[idx])}]\n"
            result_str += f"[cv score: {str(this_history['score'])}]\n"
            hp_name = []
            hp_space = self.strategy.ii[name].algo_ins.get_hp_space()
            specified_params = self.strategy.ii[name].algo_ins.specified_params
            for hp in hp_space:
                hp_name.append(hp.get_hp_name())
            for j in range(len(hp_name)):
                result_str += hp_name[j] + ' = ' + \
                    str(this_history['hps'][j]) + '\n'
            for param_name in specified_params:
                param_value = specified_params[param_name]
                result_str += param_name + ' = ' + str(param_value) + '\n'
            result_str += f"----------------\n"

        if print_res:
            print(result_str)

        return result_str

    def save_history(self, filename=None, mode='by_score', X_test=None, y_test=None):
        if filename is None:
            filename = 'history.txt'

        if mode == 'by_order':
            history = self.history_by_order
        elif mode == 'by_score':
            history = self.history_by_score
        else:
            print('ERROR: unknown param is given to [mode]')
            return 'ERROR'

        with open(filename, 'w') as f:
            if X_test is not None and y_test is not None:
                test_score = self.metric.func(y_test, self.predict(X_test))
                f.write(f"test score: {str(test_score)}\n")
                f.write('=================ensemble models=================\n')
                f.write(f"{str(self.predict_separate(X_test,y_test))}\n")
            else:
                f.write('=================ensemble models=================\n')
                for idx, this_history in enumerate(self.ensemble_info):
                    model = self.ensemble_models[idx]
                    f.write(str({'model': this_history['arm'], 'weight': self.ensemble_weights[idx],
                            'cv': this_history['score'], 'hps': model.get_param_dict(this_history['hps'])})+'\n')
            f.write('=================total history=================\n')
            f.write(str(history) + '\n')
            # f.write('=================decision values history==================\n')
            # f.write(str(self.strategy.decision_values_history) + '\n')
            f.write('=================init history=================\n')
            f.write(str(self.strategy.init_history) + '\n')
            f.write('=================thread allocation=================\n')
            f.write(str(self.strategy.threads_allocation) + '\n')
            f.write('=================tn=================\n')
            f.write(str(self.strategy.tn) + '\n')
            f.write('=================total eval time=================\n')
            f.write(str(self.strategy.total_eval_time) + '\n')


class AutomlClassifier(BaseAutoml):

    def __init__(self):
        super(AutomlClassifier, self).__init__()

        self.num_class = None
        self.task = 'clf'

        return

    def __clear(self):
        self.history_by_order = []
        self.history_by_score = []
        self.ensemble_info = []
        self.ensemble_models = []
        self.ensemble_weights = []
        self.arms = []
        self.num_class = None
        self.metric = None

    def fit(self, X, y, max_run_time=None, total_pulls=None, ensemble_strategy=None, metric='f1_macro', parallel_num=-1, stationary=True, include_algo=None, algo_config_file=None):
        if ensemble_strategy is None:
            ensemble_strategy = [4, 2]
        self.__clear()
        self.num_class = len(np.unique(y))

        return super().fit(
            X=X,
            y=y,
            max_run_time=max_run_time,
            total_pulls=total_pulls,
            ensemble_strategy=ensemble_strategy,
            metric=metric,
            parallel_num=parallel_num,
            stationary=stationary,
            include_algo=include_algo,
            algo_config_file=algo_config_file
        )

    def _predict(self, X, want_proba=False):
        if len(self.ensemble_models) != len(self.ensemble_weights):
            print('ERROR: different numbers between models and weighs')
            return

        for i in range(len(self.ensemble_models)):
            is_proba = False
            if hasattr(self.ensemble_models[i], 'predict_proba'):
                try:
                    single_result = self.ensemble_models[i].predict_proba(X)
                    is_proba = True
                except AttributeError:
                    is_proba = False
            if not is_proba:
                # convert predicted result to proba type
                single_result = self.ensemble_models[i].predict(X)
                converted_result = []
                for one_res in single_result:
                    empty = np.zeros(self.num_class)
                    empty[int(one_res)] = 1
                    converted_result.append(empty)
                single_result = np.array(converted_result)
            if i == 0:
                pred = self.ensemble_weights[i] * np.array(single_result)
            else:
                pred += self.ensemble_weights[i] * np.array(single_result)

        if want_proba:
            final_result = pred
        else:
            final_result = np.array(
                [random.choice(np.argwhere(x == np.amax(x)))[0] for x in pred])

        return final_result

    def predict(self, X):

        return self._predict(X, want_proba=False)

    def predict_proba(self, X):

        return self._predict(X, want_proba=True)


class AutomlRegressor(BaseAutoml):

    def __init__(self):

        super(AutomlRegressor, self).__init__()
        self.task = 'rgr'

        return

    def __clear(self):
        self.history_by_order = []
        self.history_by_score = []
        self.ensemble_info = []
        self.ensemble_models = []
        self.ensemble_weights = []
        self.arms = []
        self.metric = None

    def fit(self, X, y, max_run_time=None, total_pulls=None, ensemble_strategy=10, metric='mae', parallel_num=-1, stationary=True, include_algo=None, algo_config_file=None):
        self.__clear()

        return super().fit(
            X=X,
            y=y,
            max_run_time=max_run_time,
            total_pulls=total_pulls,
            ensemble_strategy=ensemble_strategy,
            metric=metric,
            parallel_num=parallel_num,
            stationary=stationary,
            include_algo=include_algo,
            algo_config_file=algo_config_file
        )

    def _predict(self, X):
        if len(self.ensemble_models) != len(self.ensemble_weights):
            print('ERROR: different numbers between models and weighs')
            return

        for i in range(len(self.ensemble_models)):

            single_result = self.ensemble_models[i].predict(X)

            if i == 0:
                pred = self.ensemble_weights[i] * np.array(single_result)
            else:
                pred += self.ensemble_weights[i] * np.array(single_result)

        return pred

    def predict(self, X):

        return self._predict(X)


if __name__ == "__main__":
    print('Automl.py')
