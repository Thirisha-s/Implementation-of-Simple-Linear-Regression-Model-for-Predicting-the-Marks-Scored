# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data and Find the values of MSE , MAE and RMSE.

## Program:
```
/*python
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S.THIRISHA.S
RegisterNumber:  212222230160
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
#segregation data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#spliting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted value
Y_pred
#displaying actual value
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(Y_test,Y_pred)
print('MSC=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:
![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/82c171fd-a331-4bc4-83b4-beb5893e0239)

![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/71bb664f-a76a-4419-9ea4-887dd625268d)

![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/832c21f5-6c17-4971-805d-09f634788907)

![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/4e91b9b3-adce-4621-8ac1-c06e9c133504)

![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/c9b33cd6-15e7-45c6-8e76-5e5e7854178c)

![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/6acfcd29-e767-489f-b539-0ef499dddbef)

![image](https://github.com/Harish2404lll/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472096/55673e75-0b52-4451-84b4-58dfce721b6b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
