from math import *

def f(mu, sigma2, x):
    return 1/sqrt(2.*pi*sigma2) * exp(-.5*(x-mu)**2/sigma2)

def update(mean1, var1, mean2, var2):
    new_mean = (var2*mean1+var1*mean2)/(var1+var2)
    new_var = 1/(1/var1+1/var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

for n in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[n], measurements_sig)
    [mu, sig] = predict(mu, sig, motion[n], motion_sig)


def __init__(self, value):
    self.value = value
    self.dimx  = len(value)
    self.dimy  = len(value[0])
    if value == [[]]:
        self.dimx = 0

def zero(self, dimx,dimy):
    # check if value dimensions
    if dimx < 1 or dimy < 1:
        raise ValueError, "Invalid size of matrix"
    else:
        self.dimx = dimx
        self.dimy = dimy
        self.value = [[0 for row in range(dimy)] for col in range(dimx)]

def identity(self, dim):
    # checkk if valid dimension
    if dim < 1:
         raise ValueError, " Invalid size of matrix"
    else:
        self.dimx = dim
        self.dimy = dim
        self.value = [[0 for row in range(dim)] for col in range(dim)]
        for i in range(dim):
            self.value[i][i] = 1

def show(self):
    for i in range(self.dimx):
        print(self.value[i])
    print()

def __add__(self, other):
    # check if correct dimensions
    if self.dimx != other.dimx or self.dimx != other.dimx:
        raise ValueError, "Matrices must be of equal dimension to a"
    else:
        # add if connect dimensions
        res = matrix([[]])
        res.zero(self.dimx, self.dimy)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[i][j] = self.value[i][j] + other.value[i][j]
        return res

def __sub__(self, other):
    # check if correct dimensions
    if self.dimx != other.dimx or self.dimx != other.dimx:
        raise ValueError, "Matrices must be of equal dimension to a"
    else:
        # subtract if connect dimensions
        res = matrix([[]])
        res.zero(self.dimx, self.dimy)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[i][j] = self.value[i][j] - other.value[i][j]
        return res

def __mul__(self, other):
    # check if correct dimensions
    if self.dimy != other.dimx:
        raise ValueError, "Matrices must be m*n and n*p to multiply"
    else:
        # multiply if connect dimensions
        res = matrix([[]])
        res.zero(self.dimx, self.dimy)
        for i in range(self.dimx):
            for j in range(other.dimy):
                for k in range(self.dimy):
                    res.value[i][j] += self.value[i][k] * other.value[k][j]
        return res

def transpose(self):
    # Compute transpose
    res = matrix([[]])
    res.zero(self.dimy, self.dimx)
    for i in range(self.dimx):
        for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        return res

def Cholesky(self, ztol=1.0e-5):
    # Computes the upper triangular Cholesky factorization of
    # a positive definite matrix.
    res = matrix([[]])
    res.zero(self.dimx, self.dimx)

    for i in range(self.dimx):
        S = sum([(res.value[k][i])**2 for k in range(i)])
        d = self.value[i][i] - S
        if abs(d) < ztol:
            res.value[i][i] = 0.0
        else:
            if d < 0.0:
                raise ValueError, "Matrix not positive-definite"
            res.value[i][i] = sqrt(d)
        for j in range(i+1, self.dimx):
            S = sum([(res.value[k][i])* res.value[k][j] for k in range(i)])
            if abs(S) < ztol:
                S = 0.0
                res.value[i][j] = (self.value[i][j] - S)/res.value[i][j]
    return res

def CholeskyInverse(self):
    # Compute inverse of matrix given its cholesky upper Triangular
    # Decomposition of matrix
    res =  matrix([])
    res.zero(self.dimx, self.dimx)

    # Backward step for inverse.
    for j in reversed(range(self.dimx)):
        tjj = self.value[j][j]
        S = sum([self.value[j][k]*res.value[j][k] for k in range(j+1,self.dimx)])
        res.value[j][j] = 1.0 / tjj**2 - S/tjj
    for i in reversed(range(j)):
        res.value[j][i] = res.value[i][j] = -sum([self.value[i]])
    return res

def inverse(self):
    aux = self.Cholesky()
    res = aux.CholeskyInverse()
    return res

def __repr__(self):
    return repr(self.value)

def filter(x,P):
    for n in range(len(measurements)):
        # measurement update
        Z = matrix([[measurements[n]]])
        y = Z - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)

        P = (I - (K * H )) * P

        # prediction
        x = (F * x ) + u
        P = F * P * F.transpose()
        

