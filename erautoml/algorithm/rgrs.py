import numpy as np
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree
from lightgbm import LGBMRegressor


FLOAT = 0
INTEGER = 1
CATEGORICAL = 2


class HyperParameter(object):

    def __init__(self, hp_name, hp_range, hp_type):
        self.hp_name = hp_name
        self.hp_range = hp_range
        self.hp_type = hp_type
        return

    @classmethod
    def int_hp(cls, hp_name, hp_range):
        return cls(hp_name, hp_range, INTEGER)

    @classmethod
    def float_hp(cls, hp_name, hp_range):
        return cls(hp_name, hp_range, FLOAT)

    @classmethod
    def categorical_hp(cls, hp_name, hp_range):
        return cls(hp_name, hp_range, CATEGORICAL)

    @property
    def param_bound(self):
        """Get lower bound and upper bound for a parameter

        Returns
        -------
        bound: tuple of int or tuple of float
            lower_bound and higher_bound are both inclusive if
            parameter's type is int or float
        """
        if self.hp_type == CATEGORICAL:
            return 0, len(self.hp_range) - 1
        else:
            return self.hp_range

    def in_range(self, value):
        """Test whether the parameter's value is in a legal range

        Parameters
        ---------
        value : str or int or float
            value of parameter

        Returns
        -------
        is_in_range: bool
            True if value is in range
        """
        if self.hp_type == CATEGORICAL:
            return 0 <= int(value) < len(self.hp_range)
        else:
            assert len(self.hp_range) == 2
            return self.hp_range[0] <= value <= self.hp_range[1]

    def get_range(self):
        if self.hp_type == CATEGORICAL:
            return 0, len(self.hp_range) - 1
        elif self.hp_type == INTEGER:
            return self.hp_range
        elif self.hp_type == FLOAT:
            return self.hp_range
        else:
            assert False, print("get hp range error!")

    def get_hp_name(self):
        return self.hp_name

    def get_hp_type(self):
        return self.hp_type

    def convert_raw_param(self, value):
        """Cast raw parameter value to certain type

        Parameters
        ----------
        value : str or int or float
            value which can be any type

        Returns
        -------
        param : str or int or float
            casted value
        """
        if self.hp_type == INTEGER:
            return int(value)
        elif self.hp_type == FLOAT:
            return float(value)
        elif self.hp_type == CATEGORICAL:
            return self.hp_range[int(value)]
        else:
            assert False

    def is_int_type(self):
        return self.hp_type == INTEGER

    def is_float_type(self):
        return self.hp_type == FLOAT

    def is_categorical_type(self):
        return self.hp_type == CATEGORICAL


class BaseAlgorithm(object):
    """
    three functions are provided in this class:
    generate_model: generate model with a set of hyper-parameters
    fit: train model
    predict: give predictions
    """

    def __init__(self, hp_space, raw_model, specified_params):
        # define the hyper-parameter space of this algorithm
        self.hp_space = hp_space
        self.raw_model = raw_model
        self.specified_params = specified_params
        self.model = None
        self.algorithm_name = ""
        self.log = ["algorithm log ---------------------------"]
        self.fitted = False
        return

    def get_param_dict(self, hp_list):
        if hp_list is not None:
            param_dict = {}
            assert len(hp_list) == len(self.hp_space)
            for value, param in zip(hp_list, self.hp_space):
                if not param.in_range(value):
                    assert False, print(self.__class__.__name__, "hp value is out of range!", param.hp_name, value)
                if param.is_categorical_type():
                    param_dict[param.hp_name] = param.hp_range[value]
                else:
                    param_dict[param.hp_name] = value
                assert hasattr(self.model, param.hp_name), print(self.__class__.__name__, "model gets invalid hp", param.hp_name)
                setattr(self.model, param.hp_name, param.convert_raw_param(value))
            param_dict.update(self.specified_params)

            return param_dict

    def apply_param_dict(self, param_dict):

        self.model = self.raw_model()

        if len(param_dict) > 0:
            for param_name in param_dict:
                param_value = param_dict[param_name]
                assert hasattr(self.model, param_name), print(self.__class__.__name__, "model gets invalid hp", param_name)
                setattr(self.model, param_name, param_value)

        return


    def get_hp_space(self):
        return self.hp_space

    def log_hp_space(self):
        self.log.append("hyper-parameter space: ")
        for hp in self.hp_space:
            lower_bound, upper_bound = hp.get_range()
            self.log.append("{}: ({}, {}), {}".format(hp.get_hp_name(), lower_bound, upper_bound, hp.get_hp_type()))

    # generate model with a set of hyper-parameters
    def generate_model(self, hp_list):

        self.model = self.raw_model()

        if hp_list is not None:
            assert len(hp_list) == len(self.hp_space)
            for value, param in zip(hp_list, self.hp_space):
                if not param.in_range(value):
                    assert False, print(self.__class__.__name__, "hp value is out of range!", param.hp_name, value)
                assert hasattr(self.model, param.hp_name), print(self.__class__.__name__, "model gets invalid hp", param.hp_name)
                setattr(self.model, param.hp_name, param.convert_raw_param(value))

        if len(self.specified_params) > 0:
            for param_name in self.specified_params:
                param_value = self.specified_params[param_name]
                assert hasattr(self.model, param_name), print(self.__class__.__name__, "model gets invalid hp", param_name)
                setattr(self.model, param_name, param_value)

        return

    # over-write fit function
    def fit(self, x, y):
        self.model = self.model.fit(x, y)
        self.fitted = True
        return

    # over-write predict function
    def predict(self, x):
        assert self.model is not None, print("model has not been generated!")
        prediction = self.model.predict(x)
        return prediction

    def predict_proba(self, x):
        assert self.model is not None, print("model has not been generated!")
        prediction = self.model.predict_proba(x)
        return prediction

    def get_algorithm_name(self):
        return self.algorithm_name

    def get_log(self):
        return self.log


class DecisionTreeRGR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):

        if hp_space is None:
            hp_space = [
                HyperParameter.categorical_hp('criterion', ('friedman_mse', 'mse', 'mae')),
                HyperParameter.int_hp('max_depth', (3, 10)),
                HyperParameter.float_hp('min_samples_leaf', (0.05, 0.2)),
            ]

        raw_model = sklearn.tree.DecisionTreeRegressor
        super(DecisionTreeRGR, self).__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "DecisionTreeRGR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return


class SVR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):
        if hp_space is None:
            hp_space = [
                HyperParameter.float_hp('tol', (1e-4, 1e-2)),
                HyperParameter.float_hp('C', (0.05, 2)),
                HyperParameter.float_hp('epsilon', (0.05, 0.2))
            ]

        raw_model = sklearn.svm.SVR
        super().__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "SVR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return


class LightGBMRGR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):

        if hp_space is None:
            hp_space = [
                HyperParameter.int_hp('num_leaves', (4, 64)),
                HyperParameter.int_hp('max_depth', (2, 16)),
                HyperParameter.float_hp('learning_rate', (0.01, 0.2)),
                HyperParameter.int_hp('n_estimators', (10, 200)),
                HyperParameter.float_hp('min_child_weight', (1e-4, 2e-3)),
                HyperParameter.int_hp('min_child_samples', (2, 40)),
                HyperParameter.float_hp('subsample', (0.5, 1)),
                HyperParameter.float_hp('colsample_bytree', (0.5, 1)),
                HyperParameter.float_hp('reg_alpha', (0, 1)),
                HyperParameter.float_hp('reg_lambda', (0, 1))
            ]

        if 'n_jobs' not in specified_params:
            specified_params['n_jobs'] = 1

        raw_model = LGBMRegressor
        super(LightGBMRGR, self).__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "LightGBMRGR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return

class KNNRGR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):

        if hp_space is None:
            hp_space = [
                HyperParameter.int_hp('n_neighbors', (1, 17)),
                HyperParameter.categorical_hp('weights', ('uniform', 'distance')),
                HyperParameter.categorical_hp('p', (1, 2))
            ]

        raw_model = sklearn.neighbors.KNeighborsRegressor
        super(KNNRGR, self).__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "KNNRGR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return

class RandomForestRGR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):
        if hp_space is None:
            hp_space = [
                # HyperParameter.categorical_hp('n_estimators', (100,)),
                HyperParameter.categorical_hp('criterion', ('mse', 'mae', 'poisson')),
                HyperParameter.int_hp('min_samples_split', (2, 20)),
                HyperParameter.int_hp('min_samples_leaf', (1, 20)),
                HyperParameter.float_hp('max_features', (0., 1.)),
                HyperParameter.categorical_hp('bootstrap', (True, False)),
            ]

        raw_model = sklearn.ensemble.RandomForestRegressor
        super().__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "RandomForestRGR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return


class AdaBoostRGR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):
        if hp_space is None:
            hp_space = [
                HyperParameter.int_hp('n_estimators', (50, 400)),
                HyperParameter.float_hp('learning_rate', (0.01, 1.)),
            ]

        raw_model = sklearn.ensemble.AdaBoostRegressor
        super().__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "AdaBoostRGR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return

class ExtraTreesRGR(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):
        if hp_space is None:
            hp_space = [
                # HyperParameter.int_hp('n_estimators', (100, 100)),
                HyperParameter.categorical_hp('criterion', ('mae', 'mse')),
                HyperParameter.int_hp('min_samples_split', (2, 20)),
                HyperParameter.int_hp('min_samples_leaf', (1, 20)),
                HyperParameter.float_hp('max_features', (0., 1.)),
                HyperParameter.categorical_hp('bootstrap', (True, False))
            ]

        raw_model = sklearn.ensemble.ExtraTreesRegressor
        super().__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "ExtraTreesRGR"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return


class GBRT(BaseAlgorithm):

    def __init__(self, hp_space=None, specified_params={}):

        if hp_space is None:
            hp_space = [
                HyperParameter.float_hp('learning_rate', (0.01, 1)),
                HyperParameter.int_hp('n_estimators', (50, 500)),
                HyperParameter.int_hp('min_samples_split', (2, 20)),
                HyperParameter.int_hp('min_samples_leaf', (1, 20)),
                HyperParameter.int_hp('max_depth', (1, 10)),
            ]

        raw_model = sklearn.ensemble.GradientBoostingRegressor
        super(GBRT, self).__init__(hp_space, raw_model, specified_params)
        self.algorithm_name = "GradientBoostingRegressor"

        self.log.append("algorithm: {}".format(self.algorithm_name))
        self.log_hp_space()

        return


if __name__ == "__main__":
    print("hello world!")