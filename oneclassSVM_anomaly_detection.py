import numpy as np
from sklearn import svm

# Train data (10000 measures with values from 18 to 40)
X_train = np.random.randint(18,41, size=(10000, 1))
# Test data (1000 new measures with values from 18 to 54)
X_test = np.random.randint(18,55, size=(1000, 1))

detect = svm.OneClassSVM(nu=0.01, kernel='linear')
# Train our network
detect.fit(X_train)

# Perform regression on samples in X_train, X_test
y_pred_train = detect.predict(X_train)
y_pred_test = detect.predict(X_test)

# Find the number of errors (outliers)
nb_error_train = y_pred_train[y_pred_train == -1].size
nb_error_test = y_pred_test[y_pred_test == -1].size

for el in y_pred_test:
    if el == -1:
        pass
        #print('Raise an event!')
        
print('Number of abnormal measurements', nb_error_train)