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