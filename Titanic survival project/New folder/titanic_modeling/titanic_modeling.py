# -*- coding: utf-8 -*-
"""

## Titanic prediction modeling
"""

import pandas as pd

df = pd.read_csv('dataset/train.csv')

print(df.head())

#0-->not survived
#1-->survived

#df.Pclass.unique()

#df.Embarked.unique()

print(df.shape)

## check missing values

print(df.isnull().sum())

print(df[df.Embarked.isnull()])

"""- Survived: 0 = No, 1 = Yes

- pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd

- sibsp: siblings / spouses aboard the Titanic

- parch: parents / children aboard the Titanic

- ticket: Ticket number

- cabin: Cabin number

- embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
"""

print(df.Survived.value_counts()) #display no of counts for each unique value

"""## plot relationship between survived feature and other feature"""

#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print(df.Fare.agg(['min','max','mean','median']))

print(df.sort_values('Fare',ascending=False)[:10])

a=[5,7,8,6,90]  #90 is a outlier
print(np.mean(a))

print(np.median(a))





plt.hist(df.Fare,bins=30)
plt.xlabel('Fare')
plt.ylabel('no of people')
plt.show()

#f[(df.Fare>30) & (df.Fare<60)]

df.Age.agg(['min','max','mean','median'])

plt.hist(df.Age,bins=35)
plt.xlabel('Fare')
plt.ylabel('no of people')
plt.show()

survived=df[df['Survived']==1].Sex.value_counts()
print(survived)

dead=df[df['Survived']==0].Sex.value_counts()
print(dead)

survived=df[df['Survived']==1].Sex.value_counts()

dead=df[df['Survived']==0].Sex.value_counts()

df_new = pd.DataFrame([survived,dead])
df_new.index = ['Survived','Dead']
df_new.plot(kind='bar',stacked=True)
plt.show()

survived=df[df['Survived']==1].Pclass.value_counts()
dead=df[df['Survived']==0].Pclass.value_counts()
df_new = pd.DataFrame([survived,dead])
df_new.index = ['Survived','Dead']
df_new.plot(kind='bar',stacked=True)
plt.show()

"""#### 1st class more likely survivied than other classes

#### 3rd class more likely dead than other classes
"""

print(df.columns)

df_1 = df.loc[:,['Pclass','Sex','Age','Fare']]

print(df_1)

"""### label encoding
### one hot encoding
"""


print(df_1.head())

#df_1.Sex=df_1.Sex.map({"male": 0, "female": 1})  #label encod

#pd.get_dummies(df_1['Sex'])

c1=pd.get_dummies(df_1['Sex'])  #one hot encoding
print(c1.head())

print(df.dtypes)

c2 = df_1.select_dtypes(exclude=['object'])  #int ,float
print(c2.head())

final_data = pd.concat([c2,c1],axis=1)
print(final_data.head())

#df_1.head()
print(final_data.head())

print(final_data.isnull().sum())

print(final_data.Age.median())

final_data.Age = final_data.Age.fillna(final_data.Age.median())

print(final_data.isnull().sum())

X = final_data.values

y = df.Survived.values

print(X.shape)

print(y.shape)

print(X.min())

#y

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.25,random_state=12)

std = StandardScaler()

Xtrain_std = std.fit_transform(xtrain)
Xtest_std = std.transform(xtest)

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
print(log.fit(Xtrain_std,ytrain))

pred = log.predict(Xtest_std)

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest,pred))