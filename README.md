# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries and read the dataframe.

2. Assign hours to X and scores to Y.

3. Implement training set and test set of the dataframe

4. Plot the required graph both for test data and training data.

5. Find the values of MSE , MAE and RMSE


## Program and Output:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: THIRISHA.S

RegisterNumber:  212222230160



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```
<img width="103" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/6cadf81c-4eaa-421f-9447-63e9491b8e0d">

```python
df.tail()
```
<img width="103" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/500b6947-d571-479c-aef7-d0c20b62b344">

```python
X=df.iloc[:,:-1].values
X
```
<img width="83" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/471f6a3d-946b-4af4-a053-d197295e0471">

```python
Y=df.iloc[:,:-1].values
Y
```
<img width="77" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/74e3ad3c-bfce-4228-af21-dc504aee34cc">

```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```
<img width="77" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/262754ef-adc1-4f7b-90e5-eba2c3811d78">

```python
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
<img width="340" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/fc969a84-5054-4867-a4f5-aed13978529a">

```python
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
<img width="359" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/1884cbe4-a46a-43db-8a99-33901dddb540">

```python
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
<img width="143" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121222763/187378d6-919e-4db2-a8c4-33a1182f7382">







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
