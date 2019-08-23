import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np  
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None)
print(df.tail())

#convert class labels into integer labels (1 = versicolor; -1 = setosa)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

