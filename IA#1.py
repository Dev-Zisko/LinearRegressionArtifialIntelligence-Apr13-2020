import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

#Cargamos libreria
boston = load_boston()

X = np.array(boston.data[:, 5])
Y = np.array(boston.target)

plt.scatter(X, Y, alpha=0.3)

#Añadiendo columna de unos para termino independiente (Formula de regresion lineal y minimos cuadrados ordinarios)
X = np.array([np.ones(506), X]).T

#Formula de regresion lineal y minimos blablabla
B = np.linalg.inv(X.T @ X) @ X.T @ Y

plt.plot([4, 9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c='red')
plt.show