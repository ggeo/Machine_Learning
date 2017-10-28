import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
sns.set()

data = pd.read_csv('./Datasets/Auto.csv', names=['mpg', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Year', 'Origin', 'name'], header=0)

# Replace strings ('?' values) in this column with value 50
data['Horsepower'] = pd.to_numeric(data['Horsepower'], errors='coerce').fillna(50).astype(int)


horsepower = data.loc[:, 'Horsepower'].to_frame()
mpg = data.loc[:, 'mpg'].to_frame()

# def f(x):
#    return np.sin(2 * np.pi * x)

# horsepower = np.random.uniform(0, 1, size=100)[:, np.newaxis]
# mpg = f(horsepower) + np.random.normal(scale=0.3, size=100)[:, np.newaxis]

# Divide data into test an train size
horsepower_train, horsepower_test, mpg_train, mpg_test = cross_validation.train_test_split(horsepower, mpg, test_size=0.5, random_state=0)

# Create linear model object and train it
linear_model_1 = linear_model.LinearRegression()
linear_model_1.fit(horsepower_train, mpg_train)

# Create polynomial object and train it
poly_model = make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression())
poly_model.fit(horsepower_train, mpg_train)

fig, ax = plt.subplots()

ax.set(xlabel='horsepower', ylabel='mpg', title='power vs mpg')
ax.scatter(horsepower, mpg, alpha=0.5, cmap='viridis', label='data')
# horsepower_test.sort(axis=0)

# Either use sort eithe use '.' in plot style
# horsepower_test.sort_index(axis=0)

ax.plot(horsepower_test, linear_model_1.predict(horsepower_test), color='cyan', linewidth=2, label='linear')
ax.plot(horsepower_test, poly_model.predict(horsepower_test), '.', markersize=14, color='green', linewidth=2, label='polynom')
ax.legend()
plt.show()
