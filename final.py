#import stuff pls
from sklearn import linear_model, preprocessing, metrics, ensemble, svm
import numpy as np

def main():
  data_set = np.genfromtxt("train.csv", delimiter=",")
  #print train_set
  data_set = np.delete(data_set, 0, axis=0)
  data_x = np.delete(data_set, 4, axis=1)
  data_y = data_set[:, 3]
  #$print data_set
  print data_x.shape
  print data_y

  num_rows = data_set.shape[0]
  '''print num_rows
  fold_size = 4 * num_rows / 5
  train_x = data_x[:fold_size, :]
  train_y = data_y[:fold_size]
  test_x = data_x[fold_size:, :]
  test_y = data_y[fold_size:]
  print train_x.shape
  print train_y.shape
  print test_x.shape
  print test_y.shape'''

  train_x = data_x
  train_y = data_y

  users = {}
  user_array = np.genfromtxt("users.csv", delimiter=",")
  user_array = np.delete(user_array, 0, axis=0)
  imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
  user_array = imp.fit_transform(user_array)
  for user in user_array:
    '''info = user[8:]
    nans = np.isnan(info)
    for i in range(len(info)):
      if nans[i]:
        info[i] = -1
    users[user[0]] = info'''
    users[user[0]] = user[4:]
    #print users[user[0]]
  #print user_array
  user_array = user_array[:, 4:]
  train_rows = num_rows
  train_cols = user_array.shape[1]
  print train_rows, train_cols

  median = np.median(user_array, axis=0)

  x_train = np.empty([train_rows, train_cols])
  i = 0
  #print users.keys()
  for line in train_x:
    user = line[2]
    #print user
    if user not in users.keys():
      x_train[i] = median
    else:
      x_train[i] = users[user]
    i+=1

  delete_these = np.where(np.all(x_train==median,axis=1))
  print delete_these
  x_train = np.delete(x_train, delete_these, axis=0)
  data_y = np.delete(data_y, delete_these, axis=0)
  #x_train = preprocessing.scale(x_train)
  
  fold_size = num_rows
  new_fold = fold_size/10
  print x_train.shape
  print x_train[new_fold:].shape
  print data_y[new_fold:fold_size].shape
  
  clf = svm.SVR()
  clf.fit(x_train[new_fold:], data_y[new_fold:fold_size])
  y_predicted = clf.predict(x_train[:new_fold])
  #clf = ensemble.GradientBoostingRegressor()
  #est = clf.fit(x_train[new_fold:], data_y[new_fold:fold_size])
  #y_predicted = est.predict(x_train[:new_fold])


  #clf = linear_model.SGDRegressor(eta0=0.00001)
  #clf.fit(x_train[new_fold:], data_y[new_fold:fold_size])
  #y_predicted = clf.predict(x_train[:new_fold])

  print data_y[:new_fold].shape
  print y_predicted.shape
  print x_train[:new_fold]
  print data_y[:new_fold]
  print y_predicted
  print metrics.mean_squared_error(data_y[:new_fold], y_predicted)


if __name__ == "__main__":
  main()
