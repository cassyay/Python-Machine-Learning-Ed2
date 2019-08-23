import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from Perceptron import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None)
print(df.tail())

#convert class labels into integer labels (1 = versicolor; -1 = setosa)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#extract sepal and petal length
X = df.iloc[0:100, [0, 2]].values

#plot
plt.scatter(X[:50, 0], X[:50, 1],
	color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
	color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

#train perceptron algorithm on iris data subset
ppn = Perceptron(eta=0.1, n_iter=10)

#plot
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1),
	ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()