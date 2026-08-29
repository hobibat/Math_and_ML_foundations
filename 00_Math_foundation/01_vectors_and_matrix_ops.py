import math

def vec_add(u, v):
    n=len(u)

    #now create a list of length n prefilled with float zeros
    w=[0.0]*n
    for i in range(n):
        w[i]= u[i]+v[i]

    return w

def scalar_mul(c, v):
    n=len(v)
    for i in range(n):
        v[i]*=c

    return v

def dot_prod(v, w):
    n=len(v)
    ans=0.0
    for i in range(n):
        ans+= v[i]*w[i]

    return ans

def Euclidean_norm(v):
    n=len(v)
    norm=0.0
    for i in range (n):
        norm+= v[i]**2

    return math.sqrt(norm) #could have returned norm**0.5 too

def mat_vec_mul(m, v):
#since each row in the result is the dot product of that row with the vector
    n=len(m) #that gives the nubmer of rows
    res_vector=[0.0]*n
    for i in range(n):
        res_vector[i]=dot_prod(m[i], v) #first row in matrix m is accessed by m[0] and so on

    return res_vector



#i still have to implement the mat_mat_mul, but not now
