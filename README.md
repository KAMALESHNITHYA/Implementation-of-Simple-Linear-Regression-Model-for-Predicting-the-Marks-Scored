# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```C
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KAMALESH R
RegisterNumber: 212223230094
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```


## Output:
## Dataset:
![image](https://github.com/user-attachments/assets/35f0d8ff-bc79-484f-b3d3-20991f4a08b7)
<br>
## Head values:
![image](https://github.com/user-attachments/assets/c76c2562-6572-46c8-bd0e-08e94288eafa)
<br>
## Tail values:
![image](https://github.com/user-attachments/assets/9bb3663b-6c66-4ffd-ab63-19dd414306b3)
<br>
## X and Y values:
![image](https://github.com/user-attachments/assets/94eb6b1b-a5bb-494a-8dce-e4f4efe28bf7)
<br>
 ## Predication values of X and Y:
![image](https://github.com/user-attachments/assets/ddd03515-1ea7-4241-bb58-127e64105d12)
<br>
 ## MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/f7de1ff2-8170-4dfa-86a7-427d13506ba1)
<br>
 ## Training Set:
![image](https://github.com/user-attachments/assets/8df133f7-fb42-4469-bd8f-f0ef432ece68)
<br>
 ## Testing Set:
![image](https://github.com/user-attachments/assets/e19193c1-1684-421a-af19-542dadaf8774)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
