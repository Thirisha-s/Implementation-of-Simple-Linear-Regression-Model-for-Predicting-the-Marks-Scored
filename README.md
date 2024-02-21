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

```python
Program to implement the simple linear regression model for predicting the marks scored.

Developed by:   THIRISHA S
RegisterNumber:  212222230160

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```

```python
df.tail()
```

```python
X= df.iloc[:,:-1].values
X
```
```python
Y = df.iloc[:,1].values
Y
```

```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
```

```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```

```python
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


```python
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```

```python
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

#### Head Values
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/da3a72b5-ca2e-4e21-b29c-8b7a4760b339)


#### Tail Values
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/bce0af5b-b551-48e7-8855-bfe729b73426)


#### X and Y values
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/61bd07bb-344b-4942-928d-919ee370c373)

![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/d3852f8a-adb0-41c6-96fb-b23940a9c13f)

####  Prediction Values
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/d0680239-c010-4022-8a5b-263237e8362c)


#### MSE,MAE and RMSE
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/c045d1e1-edb1-471e-a6eb-1b99568c9fdc)


#### Training Set
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/003fafd5-5230-4268-9f42-8777dbf6181f)


#### Testing Set
![image](https://github.com/Leann4468/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121165979/54dc3f5e-a157-4740-ad2e-b36bf9591282)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
