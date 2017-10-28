import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

X_train = np.random.randint(18,41, size=(200, 2))
X_test = np.random.randint(18,41, size=(20, 2))
X_outliers = np.random.randint(18,70, size=(10, 2))

detect = svm.OneClassSVM(kernel='rbf', gamma=0.01)
detect.fit(X_train)

y_pred_train = detect.predict(X_train)
y_pred_test = detect.predict(X_test)
y_pred_outliers = detect.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


# plot the line, the points, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(10, 70, 700), np.linspace(10, 70, 700))
Z = detect.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# fig size
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 70), cmap=plt.cm.RdGy)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], cmap='Blues')

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='green')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blue')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold')
plt.axis('tight')
plt.xlim((10, 70))
plt.ylim((10, 70))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()