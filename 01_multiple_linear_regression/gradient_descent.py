import numpy as np
np.random.seed(8) #freeze generated random numbers so they never change on reruns

def get_cost(X, y, w, b):
    m=len(y)
    y_hat=np.dot(X, w) +b
    cost=np.sum((y_hat -y)**2)
    #this is nuts, i didn't use a loop for getting the sum too!

    cost/=2*m
    return cost



def numeric_gradient(X_train, y_train, w_init, b_init, alpha):
    eps=1e-7

    w=w_init.copy()
    b=b_init

    n=len(X_train[0])
    dj_dw=np.zeros(n)
    
    for j in range(n):
        #i will use the cost function to compute the value of the function obviously
        og_j= get_cost(X_train, y_train, w, b)

        w[j]+=eps
        changed_jw= get_cost(X_train, y_train, w, b)
        dj_dw[j]= (changed_jw - og_j)/eps

        #and don't forget to reset w[j] for the next partial derivative
        w[j]-=eps


    changed_jb= get_cost(X_train, y_train, w, b+eps)
    dj_db= (changed_jb - get_cost(X_train, y_train, w, b))/eps

    w=w- alpha*dj_dw
    b=b- alpha*dj_db

    return w, b

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

    for epoch in range(iters):

        w, b= analytical_gradient(X_train, y_train, w, b, alpha)

        #for debugging
        if epoch % 100==0:
            cost=get_cost(X_train, y_train, w, b)
            print(f" cost on {epoch}th iteration is: {cost: .4f}")

    return w, b

    

def numeric_gradient_descent(X_train, y_train, w_init, b_init, iters):
    #so the aim now is that i will compute the gradient for w and b
    #which we concluded that it's the matrix multiplication for the errors vector and transpose of x
    #and i would do this for thousand times and update w
    w=w_init.copy()
    b=b_init

    for epoch in range(iters):

        w, b= numeric_gradient(X_train, y_train, w, b, 0.01)

        #for debugging
        if epoch % 100==0:
            cost=get_cost(X_train, y_train, w, b)
            print(f" cost on {epoch}th iteration is: {cost: .4f}")

    return w, b  




m=10
n=6
#m data points where each point is a vector of n features
X_train= np.random.randn(m, n) #random matrix of mxn

#these are just to get the y_train values from the random x values
w_true = np.random.randn(n)
b_true = np.random.randn()



#ture vector of parameters
y_train =np.dot(X_train, w_true) +b_true


#initial vector of parameters
w_init = np.random.randn(n)

#initial bias
b_init = np.random.randn()



w_final, b_final= gradient_descent(X_train, y_train, w_init, b_init, 0.01, 1500)

for i in range(n):
    print(f" weight no.{i+1}: {w_final[i]: .4f}")
print(f"final bias is: {b_final: .4f}")
print("\n \n")
#________________________________________________________________

w_num, b_num= numeric_gradient_descent(X_train, y_train, w_init, b_init, 1500)
for i in range(n):
    print(f" weight no.{i+1}: {w_num[i]: .4f}")
print(f"{b_num: .4f}")

