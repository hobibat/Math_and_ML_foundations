import numpy as np

#______________________________________________________________
import os
import sys
import importlib.util

# 1. Resolve path to the normal equation file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
normal_eq_path = os.path.join(
    project_root, "00_Math_foundation", "03_normal_equation.py"
)

# 2. Load the module dynamically (handles file names starting with numbers)
spec = importlib.util.spec_from_file_location(
    "normal_equation_module", normal_eq_path
)
normal_eq_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normal_eq_module)

# 3. Pull the function
normal_equation = normal_eq_module.normal_equation



import importlib.util
from pathlib import Path
# Locate the file dynamically
target_file = Path(__file__).resolve().parent.parent / "00_Math_foundation" / "02_numerical_gradient.py"
spec = importlib.util.spec_from_file_location("numerical_gradient_module", target_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Pull the function
numerical_descent = module.numeric_gradient_descent

#______________________________________________________________________

def z_score_normalization(X_train):
    #it's the x1 minus mu over the deviation
    mu= np.mean(X_train, axis=0) #mu will be an nx1 column matrix that has the mean of each column
    sigma= np.std(X_train, axis=0)

    normalized_x= (X_train - mu)/sigma
    return normalized_x
#______________________________________________________________________

np.random.seed(8) #freeze generated random numbers so they never change on reruns

def get_cost(X, y, w, b):
    m=len(y)
    y_hat=np.dot(X, w) +b
    cost=np.sum((y_hat -y)**2)
    #this is nuts, i didn't use a loop for getting the sum too!

    cost/=(2*m)
    return cost

#___________________________________________________________________________

def analytical_gradient(X_train, y_train, w_init, b_init, alpha):
    m=len(X_train)
    w=w_init.copy()
    b=b_init

    loss= (np.dot(X_train, w) + b) -y_train

    #now calculate the gradiant
    dj_dw = (X_train.T @ loss)/m 
    dj_db = np.sum(loss)/m #its derivative is just the sum of errors as we concluded

    #update the weight and bias
    w-= alpha*dj_dw
    b-= alpha*dj_db

    return w, b


def gradient_descent(X_train, y_train, w_init, b_init, alpha, iters):
    #so the aim now is that i will compute the gradient for w and b
    #which we concluded that it's the matrix multiplication for the errors vector and transpose of x
    #and i would do this for thousand times and update w
    w=w_init.copy()
    b=b_init
    cost_history=[0.0]*iters

    for epoch in range(iters):

        w, b= analytical_gradient(X_train, y_train, w, b, alpha)
        cost_history[epoch]=get_cost(X_train, y_train, w, b)

    return w, b, cost_history




m=10
n=6
#m data points where each point is a vector of n features
X_train= np.random.randn(m, n) #random matrix of mxn

#these are just to get the y_train values from the random x values
w_true = np.random.randn(n)
b_true = np.random.randn()



#ture vector of parameters
y_train =np.dot(z_score_normalization(X_train), w_true) +b_true


#initial vector of parameters
w_init = np.random.randn(n)

#initial bias
b_init = np.random.randn()



w_final, b_final, cost_history= gradient_descent(
     z_score_normalization(X_train)
     , y_train, w_init, b_init, 0.1, 1500
     )#i changed alpha to a bigger value cuz we use the normalized x now, so we don't want to make the step such a small value
      #cuz normalization makes the function run way faster


#________________________________________________________________

w_num, b_num= numerical_descent( 
    z_score_normalization(X_train)
    , y_train, w_init, b_init, 1500
    )


# Pass training data into normal_equation
w_normal, y_hat_normal, e_normal = normal_equation(
     z_score_normalization(X_train),
     y_train
     )

# Compare with gradient descent weights
print("Normal Equation Weights:\n", w_normal[:-1])
print("Gradient Descent Weights:\n", w_final)
print("NUmerical Gradient Descent Weights:\n", w_num)


print("\n \nNormal Equation bias: \n", w_normal[-1])
print("Gradient Descent bias:\n", b_final)
print("Numerica Gradient Descent bias:\n", b_num)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(cost_history, color="royalblue", linewidth=2)
plt.title("Cost vs. Iterations (Learning Curve)", fontsize=14)
plt.xlabel("Iteration / Epoch", fontsize=12)
plt.ylabel("Cost J(w, b)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()