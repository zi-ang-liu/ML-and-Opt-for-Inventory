from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import matplotlib.pyplot as plt

# Define the true function that we want to estimate
def f(x):
    return x * np.sin(x)

X = np.linspace(0, 10, 200)
y = f(X)


# sample 5 points as training data
X_train = np.random.choice(X, 5)
y_train = f(X_train)

# fit the Gaussian Process model
gp = GaussianProcessRegressor()
gp.fit(X_train.reshape(-1, 1), y_train)

mean, std = gp.predict(X.reshape(-1, 1), return_std=True)

# aquisition function
lb = mean - 10 * std


# plot the function, the prediction and the 95% confidence interval
plt.plot(X, y, label='True function: $f(x) = x\,\sin(x)$')
plt.plot(X, mean, 'k--', label='GP mean')
plt.fill_between(X, mean - 1.96 * std, mean + 1.96 * std, alpha=0.2, label=r'95% confidence interval')
plt.scatter(X_train, y_train, label='Training data')
plt.plot(X, lb, 'r--', label='Lower Confidence Bound')
plt.legend()
plt.savefig('gp-example.pdf')


