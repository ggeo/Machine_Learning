import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation

sns.set()

data = pd.read_csv("./Datasets/Advertising.csv", names=['TV', 'Radio', 'Newspaper','Sales'], header=0)

fig = plt.figure()
ax = plt.axes()

tv = data.loc[:, 'TV'].to_frame()
sales = data.loc[:, 'Sales'].to_frame()

tv_train, tv_test, sales_train, sales_test = cross_validation.train_test_split(tv, sales, test_size=0.4, random_state=0)

# OR -- using manually train and test data ---- #

# Training data
#tv_train = data.loc[0:100,'TV'].to_frame()
#sales_train = data.loc[0:100,'Sales'].to_frame()

# Test data 
#tv_test = data.loc[100:,'TV'].to_frame()
#sales_test = data.loc[100:,'Sales'].to_frame()

# ---------------------------------------------- #

# Create linear regression object
linear_regr = linear_model.LinearRegression()

# Train the model using the training sets
linear_regr.fit(tv_train, sales_train)

# The coefficients
print('Coefficients: \n', linear_regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"% np.mean((linear_regr.predict(tv_test) - sales_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linear_regr.score(tv_test, sales_test))

ax.set(xlim=(-1, 300), ylim=(0, 30), xlabel='TV', ylabel='Sales', title='TV sales')

ax.scatter(tv_test, sales_test, alpha=0.5, cmap='viridis')

ax.plot(tv_test, linear_regr.predict(tv_test), color='red', linewidth=2)

dy=2
plt.errorbar(tv_test.values, sales_test.values, yerr=dy, fmt='o', color='blue',
             ecolor='gray', elinewidth=1, capsize=0);

plt.show()