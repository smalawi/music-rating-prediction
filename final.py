from sklearn import linear_model, preprocessing, metrics, ensemble, svm, tree
from sklearn.cross_validation import cross_val_score
import numpy as np
import time
import math

def main():
  data_set = np.genfromtxt("train.csv", delimiter=",")
  data_set = np.delete(data_set, 0, axis=0)
  data_x = np.delete(data_set, 4, axis=1)
  data_y = data_set[:, 3]

  num_rows = data_set.shape[0]

  train_x = data_x
  
  imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

  words = {}
  word_array = np.genfromtxt("words.csv", delimiter=",")
  word_array = np.delete(word_array, 0, axis=0)
  word_array = np.delete(word_array, [2, 3, 4, 19, 20], axis=1)
  word_array = imp.fit_transform(word_array)
  for word in word_array:
    artist = word[0]
    user = word[1]
    words[(artist, user)] = word[2:]
  word_array = word_array[:, 2:]

  users = {}
  user_array = np.genfromtxt("users.csv", delimiter=",")
  user_array = np.delete(user_array, 0, axis=0)
  user_array = np.delete(user_array, [1, 3, 4, 5, 6, 7], axis=1)
  user_array = imp.fit_transform(user_array)
  for user in user_array:
    users[user[0]] = user[1:]
  user_array = user_array[:, 1:]
  
  train_rows = num_rows
  user_cols = user_array.shape[1]
  word_cols = word_array.shape[1]
  train_cols = user_cols + word_cols

  user_median = np.median(user_array, axis=0)
  word_median = np.median(word_array, axis=0)
  
  new_line = np.empty([1, train_cols])
  x_train = np.empty([train_rows, train_cols])
  i = 0
  for line in data_x:
    artist = line[0]
    user = line[2]
    x_train[i, 0:user_cols] = users.get(user, user_median)
    x_train[i, user_cols:] = words.get((artist, user), word_median)
    i += 1
  
  print "Feature array complete"

  start = time.time()
  
  clf = ensemble.RandomForestRegressor(n_estimators=100, max_depth = 40, max_features='log2')
  print math.sqrt(-1*np.mean(cross_val_score(clf, x_train, data_y, scoring='mean_squared_error')))
  
  end = time.time()
  print "Time:", end - start

if __name__ == "__main__":
  main()
