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