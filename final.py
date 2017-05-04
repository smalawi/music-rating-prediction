#import stuff pls
from sklearn import linear_model, preprocessing
import numpy as np

def main():
  data_set = np.genfromtxt("train.csv", delimiter=",")
  #print train_set
  data_set = np.delete(data_set, 0, axis=0)
  data_x = np.delete(data_set, 4, axis=1)
  data_y = data_set[:, 4]
  #$print data_set
  print data_x.shape
  print data_y

  num_rows = data_set.shape[0]
  print num_rows
  fold_size = 4 * num_rows / 5
  train_x = data_x[:fold_size, :]
  train_y = data_y[:fold_size]
  test_x = data_x[fold_size:, :]
  test_y = data_y[fold_size:]
  print train_y.shape[0]
  print test_y.shape[0]
  
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
    users[user[0]] = user[8:]
    print users[user[0]]
  #print user_array

if __name__ == "__main__":
  main()
