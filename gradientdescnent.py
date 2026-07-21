import numpy as np

def sigmoid(s):
  return 1 / (1 + np.exp(-s))


def J(X, y, w, b):
    m = X.shape[0]
    g = sigmoid((np.dot(X, w) + b))
    return (- (1 / m) * np.sum(y * np.log(g) + (1 - y) * np.log(1 - g)))

def leftside(X, y, w, b):
    m = X.shape[0]
    g = sigmoid((np.dot(X, w) + b))
    dwj = (1 / m) * np.dot(X.T, (g - y))
    dbj = (1 / m) * np.sum((g - y))
    return dwj, dbj

def rightside(X, y, w, b):
    dwj = np.zeros_like(w)
    dbj = 0
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += (1e-8)
        w_minus[i] -= (1e-8)

        dwj[i] = ((J(X, y, w_plus, b)) - (J(X, y, w_minus, b))) / (2 *(1e-8))

    dbj = ((J(X, y, w, (b +(1e-8)))) - (J(X, y, w, (b -(1e-8))))) / (2 *(1e-8))
    return dwj, dbj


m, n = 200, 2
X = np.random.rand(m, n)
y = (np.sum(X, axis=1) > 1).astype(int)
w = np.random.randn(n)
b = np.random.randn()

dw_j, db_j = leftside(X, y, w, b)
dw_j1, db_j1 = rightside(X, y, w, b)

print(dw_j ,'and' ,dw_j1 , 'are dw')
print('diff of them is:', np.linalg.norm(dw_j - dw_j1))

print('\nand' ,db_j, 'and' , db_j1 , 'are db')
print('diff is:', abs(db_j - db_j1))
print('\nthe diffs are small')
