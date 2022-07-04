from .algorithm.clfs import AdaBoost, Bagging, BernoulliNB, DecisionTree, ExtraTree, ExtraTrees, GaussianNB, GaussianProcess, KNeighbors, \
    LightGBM, LinearSVC, LogisticRegression, MultinomialNB, NuSVC, PassiveAggressive, Perceptron, QDA, RadiusNeighbors, RandomForest, Ridge, SGD, \
        SVC, LDA, GBDT

from .algorithm.rgrs import SVR, DecisionTreeRGR, LightGBMRGR, KNNRGR, \
                                RandomForestRGR, AdaBoostRGR, ExtraTreesRGR, \
                                GBRT


all_clf_class = [AdaBoost, Bagging, BernoulliNB, DecisionTree, ExtraTree, ExtraTrees, GaussianNB, GaussianProcess, KNeighbors, LightGBM, LinearSVC, \
    LogisticRegression, MultinomialNB, NuSVC, PassiveAggressive, Perceptron, QDA, RadiusNeighbors, RandomForest, Ridge, SGD, SVC, LDA, GBDT]
all_rgr_class = [SVR, DecisionTreeRGR, LightGBMRGR, KNNRGR, 
                                RandomForestRGR, AdaBoostRGR, ExtraTreesRGR,
                                GBRT]
