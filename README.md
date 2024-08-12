# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KAMALESH R
RegisterNumber:212223230094

```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
![image](https://github.com/user-attachments/assets/00ba21a4-a46d-43f3-8642-f3ca2d84f42e)
<br>
![image](https://github.com/user-attachments/assets/dc00a645-8b3d-496f-a10b-595a067dbb12)
<br>
![image](https://github.com/user-attachments/assets/cc767ef2-45e2-4c41-8fc8-aa2b67464775)
<br>
![image](https://github.com/user-attachments/assets/0e642af6-5096-451e-88df-afe17596dab3)
<br>
![image](https://github.com/user-attachments/assets/0e1011fb-d8d0-44e2-bc6e-507be41a478e)
<br>
![image](https://github.com/user-attachments/assets/10ffc128-ed99-4a5b-95dd-493f9c302cee)
<br>
![image](https://github.com/user-attachments/assets/636fce05-0cc5-468d-b085-7d739b232264)
<br>
![image](https://github.com/user-attachments/assets/e898e77f-ade5-4d1f-967b-209f8c5cf478)
<br>
![image](https://github.com/user-attachments/assets/a247c68a-d8d7-424f-87f8-b648ebcd44c9)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
