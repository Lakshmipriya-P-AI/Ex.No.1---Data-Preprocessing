# Ex.No.1---Data-Preprocessing

## AIM: 
To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
df.head()
le=LabelEncoder()
df["CustomerId"]=le.fit_transform(df["CustomerId"])
df["Surname"]=le.fit_transform(df["Surname"])
df["CreditScore"]=le.fit_transform(df["CreditScore"])
df["Geography"]=le.fit_transform(df["Geography"])
df["Gender"]=le.fit_transform(df["Gender"])
df["Balance"]=le.fit_transform(df["Balance"])
df["EstimatedSalary"]=le.fit_transform(df["EstimatedSalary"])
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)
df.duplicated()
print(df['Exited'].describe())
scaler= MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
x_train,x_test,y_train,x_test=train_test_split(X,Y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT
### Printing first five rows and cols of given dataset:
![229359845-cb939621-0459-43ec-89fd-a6f64954d8b9](https://user-images.githubusercontent.com/93427923/229425546-563d0c66-42b5-42c1-a356-b4370b37e9c3.png)

### Seperating x and y values:
![229359871-b0f2390f-7c2c-4475-9c06-77f9e02443d9](https://user-images.githubusercontent.com/93427923/229426419-6ece0ea6-f7c8-4222-827b-ed14b6c94ab3.png)


### Checking NULL value in the given dataset:
![229359905-7242ba39-daef-4fbe-b5a8-68b3fecfbc62](https://user-images.githubusercontent.com/93427923/229426335-a8f1f6c2-58d1-4aa0-abc8-321ac895021d.png)

### Printing the Y column along with its discribtion:
![229360039-2e50d2bf-3ed3-49fa-bcb2-eb6b4abc2973](https://user-images.githubusercontent.com/93427923/229426455-76b81b28-5d87-4ed6-a580-6c7771be146c.png)

### Applying data preprocessing technique and printing the dataset:
![229360121-85463e88-3fa5-4d1f-825f-4acddd762314](https://user-images.githubusercontent.com/93427923/229425939-b2713fc8-51f4-4f1d-9258-66d9e585fe12.png)

### Printing training set:
![229360208-dd9c0e8f-f1b2-4b13-8124-c8dbac7ad485](https://user-images.githubusercontent.com/93427923/229426024-9f0f64e4-b48e-474b-b878-66954f53cbe8.png)

### Printing testing set and length of it:
![229360244-28e2fa80-b67b-4431-aff2-1df167281f9c](https://user-images.githubusercontent.com/93427923/229426135-f2936ca5-cdbc-435f-b992-6d311b8d1818.png)


## RESULT
Hence the data preprocessing is done using the above code and data has been splitted into trainning and testing data for getting a better model.
