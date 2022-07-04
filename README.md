# ERUCB-based AutoML tool

ERUCB-based AutoML tool is a Python Automated Machine Learning tool based on the CASE framework and ERUCB algorithm(about the framework and the algorithm please read this [paper](https://ieeexplore.ieee.org/abstract/document/9477007 "paper")).

# Installation

```
pip install -e .
```

# Usage

The tool can be used with Python code.
You can use class AutomlClassifier for classification tasks:
```
from erautoml import AutomlClassifier
```
or use class AutomlRegressor for regression tasks:
```
from erautoml import AutomlRegressor
```

# Examples

## Classification


```
import time
from sklearn.metrics import f1_score
from erautoml import AutomlClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load dataset
dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33, random_state=42)

# auto learn
automl = AutomlClassifier()
start_time = time.time()
automl.fit(X_train, y_train,  total_pulls=200)
automl.save_history('history_clf.txt')
duration = time.time() - start_time

# predict
y_pred = automl.predict(X_test)
score = f1_score(y_test, y_pred, average='macro')
print(f"Get test F1-macro of {score} in {duration} secs.")
```

## Regression

```
import time
from sklearn.metrics import mean_absolute_error
from erautoml import AutomlRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load dataset
dataset = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33, random_state=42)

# auto learn
automl = AutomlRegressor()
start_time = time.time()
automl.fit(X_train, y_train, total_pulls=200)
automl.save_history('history_rgr.txt', mode='by_order')
duration = time.time() - start_time

# predict
y_pred = automl.predict(X_test)
score = mean_absolute_error(y_test, y_pred)
print(f"Get test MAE of {score} in {duration} secs.")
```

# Configuration

## Classification
Default configurations of algorithms and spaces of hyperparameters are defined in ./erautoml/algorithm/clfs.py  
You can edit the file directly for convinience, or you can pass a parameter `algo_config_file` to AutomlClassifier().
You can use method `dump_ini()` of AutomlClassifier() to see an example of `algo_config_file`, this method will dump hyperparameter space info from ./erautoml/algorithm/clfs.py to .ini file.
You should know that by passing parameter `algo_config_file`, you can only set the spaces of hyperparameters, you can only add new base algorithms by editing ./erautoml/algorithm/clfs.py.

## Regression
Default configurations of algorithms and spaces of hyperparameters are defined in ./erautoml/algorithm/rgrs.py  
You can edit the file directly for convinience, or you can pass a parameter `algo_config_file` to AutomlRegressor().  
You can use method `dump_ini()` of AutomlRegressor() to see an example of `algo_config_file`, this method will dump hyperparameter space info from ./erautoml/algorithm/rgrs.py to .ini file.
You should know that by passing parameter `algo_config_file`, you can only set the spaces of hyperparameters, you can only add new base algorithms by editing ./erautoml/algorithm/rgrs.py.

# Citing ERUCB-based AutoML tools

If you use ERUCB-based AutoML tool in a scientific publication, please consider citing the following paper:

Yi-Qi Hu, Xu-Hui Liu, Shu-Qiao Li, Yang Yu.  [Cascaded Algorithm Selection with Extreme-Region UCB Bandit](https://ieeexplore.ieee.org/abstract/document/9477007 "paper"). **IEEE Transactions on Pattern Analysis and Machine Intelligence**, in press.
