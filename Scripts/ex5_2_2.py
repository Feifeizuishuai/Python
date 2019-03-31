# exercise 5.2.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import numpy as np

from sklearn.preprocessing import PolynomialFeatures


# Use dataset as in the previous exercise
N = 100
X = np.random.uniform(-3, 3, size=N).reshape(-1,1)
eps_mean, eps_std = 0, 0.1
eps = np.array(eps_std*np.random.randn(N) + eps_mean).reshape(-1,1)
w0 = -0.5
w1 = 0.01
w2 = 0.05
y = w0 + w1*X + w2*X**2 + eps
y_true = y - eps

poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)

# Fit ordinary least squares regression model
model = lm.LinearRegression(fit_intercept=False)
model = model.fit(X2,y)
# Compute model output:
y_est = model.predict(X2)
# Or equivalently:
#y_est = model.intercept_ + X @ model.coef_


# Plot original data and the model output
f = figure()
plot(X,y,'.')
plot(X,y_true,'*')
plot(X,y_est,'.')
xlabel('X'); ylabel('y')
legend(['Training data', 'Data generator', 'Regression fit (model)'])

show()

print('Ran Exercise 5.2.2')