import numpy as np


def normal_equation(X_train, y_train):

    m=len(X_train)
    bias= np.ones((m, 1)) #remeber to addd that 1 so it becomes a column matrix not just a vector

    #now, i will use np.c_ which stands for concatinate along columns
    A= np.c_[X_train, bias]
    w=np.linalg.inv(A.T @ A) @ (A.T @ y_train)

    y_hat= A @ w
    e=y_train - y_hat

    return w, y_hat, e


if __name__ == "__main__":
    np.random.seed(8)

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
    w, y_hat, e = normal_equation(X_train, y_train)

    print("weights:")
    for i in range(n):
        print(f"{w[i]: .4f}")

    print(f"\n bias: {w[-1]: .4f}")

    print("\n errors: ")
    for i in range(m):
        print(f"{e[i]: .4f}")

    print("\n predictions: ")
    for i in range(m):
        print(f"{y_hat[i]: .4f}")


