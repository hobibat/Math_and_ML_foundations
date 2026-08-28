import numpy as np

x_train=np.array([2.0, 4.0, 6.0])
y_train=np.array([1.0, 3.0, 5.0])


    #we need to know that we won't need it, it's just a confirmation or incase if we want to compare models
def get_cost(x, y, w, b, m):
    #first calculate the cost
    j_wb=0 #this is the inital value of the cost
    for i in range(m):
        f_wb = w*x[i] +b
        j_wb+= (f_wb -y[i])**2

    return j_wb/ (2*m)

def gradient_descent(x, y, wi, bi, m, alpha, iters):
    w=wi
    b=bi

    for i in range(iters):
        d_jw=0.0
        d_jb=0.0
        for j in range(m):
            #calc the derivative
            f_wb= w*x[j] +b
            d_jw+= (f_wb - y[j])*x[j] 
            d_jb+= (f_wb - y[j])

        d_jw/=m    
        d_jb/=m    
        w = w - alpha* d_jw
        b= b- alpha* d_jb
        #these are gonna be updated thousand time to reach the only min point

        #now i will confirm that this will make the cost decrease
        
        if i%100 ==0 or i==0 :
            cost=get_cost(x_train, y_train, w, b, m)
            print(f"{cost: .4f}")
            #gets printed on each 100 iterations and the first
    return w, b


#we first assume values for w and b then keep updating them
wi=1.0
bi=0.0
m=len(x_train) #no. of data points
alpha=0.01 #alpha is something of our choice to minimize el lelah
iters= 1000


final_w, final_b = gradient_descent(x_train, y_train, wi, bi, m, alpha, iters)
print(
    f"final parameters are: \n w={final_w: .4f} \n b={final_b: .4f}"
    )
