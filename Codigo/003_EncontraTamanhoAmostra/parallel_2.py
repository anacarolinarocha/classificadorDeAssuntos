import numpy as np
import pandas as pd


teste = [['NY', 'blue', 'Steak', 30, 165, 4.6],
       ['TX', 'green', 'Lamb', 2, 70, 8.3],
       ['FL', 'red', 'Mango', 12, 120, 9.0],
       ['AL', 'white', 'Apple', 4, 80, 3.3],
       ['AK', 'gray', 'Cheese', 32, 180, 1.8],
       ['TX', 'black', 'Melon', 33, 172, 9.5],
       ['TX', 'red', 'Beans', 69, 150, 2.2]]

df = pd.DataFrame(teste)

teste[:,0]

df[:,0]

type(teste)
type(df)

teste = [[30, 165, 4.6],
       [2, 70, 8.3],
       [12, 120, 9.0],
       [4, 80, 3.3],
       [32, 180, 1.8],
       [33, 172, 9.5],
       [69, 150, 2.2]]


type(x)
x = np.random.rand(3,4)
x
x[:,0]