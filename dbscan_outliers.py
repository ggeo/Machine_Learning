import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#X = np.array([1,2,2.2,8.1,3,1.5,6,1.7,2.3,1.5,2.9,3.1,2.3,2.5,3,1.7,1.9,7,2,2,3.1,1.7,
#              5,2.5, 1.5, 2.3,3,1.7,1.8,1.1,1.2,1.3,1.4,2.5,2.2,2.3,3.1,5.7, 7,2,1.4,8,
#             2.3, 3.1,6,1,7,2,7,1,8,2.3,7,2,6,1,8,2,9,10,1,2,8,3,2,1,7,1,2,1,8,1,2,1,9,
#              3,2,1,8,1,1,7,8,9,1,8,9])

some_data = np.random.normal(5, 0.5, size=100)
outliers = np.array([1.4, 10.2, 8.9, 5,6.5, 1.5, 1.8, 8, 7.5, 7.3])
X = np.hstack((some_data, outliers))
X = X.reshape(-1, 1)
#X = StandardScaler().fit_transform(X)
# set n_jobs=-1 in order to use all cpu cores
db = DBSCAN(eps=0.3, min_samples=15, n_jobs=1).fit(X) 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black color used for outliers.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy, 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy, 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
print('\nOutliers: \n', X[class_member_mask & ~core_samples_mask])