# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Nithilan S
RegisterNumber: 212223240183
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
#read csv file
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
# Segregating data to variables
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
#displaying predicted values
print("Nithilan S")
print("212223240108")
print(y_pred)
#displaying actual values
print("Nithilan S")
print("212223240108")
y_test
#graph plot for training data
print("Nithilan S")
print("212223240108")
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
print("Nithilan S")
print("212223240108")
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#find mae,mse,rmse
print("Nithilan S")
print("212223240108")
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### df.head
![image](https://github.com/user-attachments/assets/b27b0aa4-0745-406a-b558-a3ff9592aedd)
### df.tail()
![image](https://github.com/user-attachments/assets/c95ff9a7-0e1e-4961-a3e2-6dc27bf8fc99)
### Array value of X
![image](https://github.com/user-attachments/assets/52f1ba03-5763-4923-961d-14a42e3319a4)
### Array value of Y
![image](https://github.com/user-attachments/assets/74d43b46-bf24-4e8e-a43e-d67ad637b750)
### Values of Y prediction
![image](https://github.com/user-attachments/assets/2c844a37-75b6-4cde-a297-dbbc12a73ffd)
### Array values of Y test
![image](https://github.com/user-attachments/assets/7428a5f3-abda-468a-a7be-3051a3d2bf62)
### Training Set Graph
<img src="https://github.com/user-attachments/assets/fe424c16-5093-4400-861e-bf2529f805b5" width="400"/>
### Test Set Graph
<img src="https://github.com/user-attachments/assets/0909cbc5-8f84-41cf-860e-836c1428d3b3
" width="400"/>
### Values of MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/33a80675-68c4-4182-99bd-74a7404876ae)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
