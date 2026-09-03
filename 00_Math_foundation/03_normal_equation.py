import numpy as np
np.random.seed(8)

def normal_equation(X_train, y_train):

    w=np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)

    y_hat= X_train @ w
    e=y_train - y_hat

    return w, y_hat, e




m=10
n=6
#m data points where each point is a vector of n features
X_train= np.random.randn(m, n) #random matrix of mxn

#these are just to get the y_train values from the random x values
w_true = np.random.randn(n)


# Generate y_train with some noise added so error isn't strictly 0
noise = np.random.randn(m) * 0.1
#ture vector of parameters
y_train =np.dot(X_train, w_true)+ noise




#______________________________________________________________________
w, y_hat, e= normal_equation(X_train, y_train)

print("weights:")
for i in range(n):
    print(f"{w[i]: .4f}")

print("\n errors: ")
for i in range(m):
    print(f"{e[i]: .4f}")

print("\n predictions: ")
for i in range(m):
    print(f"{y_hat[i]: .4f}")


