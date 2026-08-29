import numpy as np



def cost(x, y, w, b):
    m=len(x) #no. of data points
    j_wb=0
    for i in range(m):
        f_wb= np.dot(x[i], w)+b
        j_wb+=(f_wb- y[i])**2

    return j_wb/(2*m)




#so the first step is to initialize w and b with arbitary values 


x_train = np.array([
    [2104.0, 5.0, 1.0, 45.0],
    [1416.0, 3.0, 2.0, 40.0],
    [1534.0, 3.0, 2.0, 30.0],
    [852.0,  2.0, 1.0, 36.0],
    [1940.0, 4.0, 2.0, 15.0],
    [2300.0, 4.0, 2.0, 10.0],
    [1200.0, 3.0, 1.0, 25.0],
    [3000.0, 5.0, 2.0, 8.0]
])

y_train = np.array(
    [460.0, 232.0, 315.0, 178.0, 410.0, 520.0, 290.0, 680.0]
    )

w_init = np.array([0.2, 10.0, -15.0, -1.0])
b_init = 50.0

ans=cost(x_train, y_train, w_init, b_init)
print(f"the cost is : {ans: .4f}")