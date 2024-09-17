import numpy as np

A = np.array([[0.52, 0.20, 0.25],
              [0.30, 0.50, 0.20],
              [0.18, 0.20, 0.55]])


b = np.array([4800, 5810, 5690])

tol = 1e-5
max_iter = 150

def jacobi(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    errores = []
    
    for iteration in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        error = np.max(np.abs(x_new - x))
        errores.append(error)
        x = np.copy(x_new)
        if error < tol:
            print(f"Convergencia alcanzada en la iteración {iteration+1}")
            break
    
    return x, errores

x, errores = jacobi(A, b, tol, max_iter)

print(f"Soluciones: {x}")
print("Errores por iteración:")
for i, error in enumerate(errores):
    print(f"Iteración {i+1}: error = {error}")
