import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# Data

df = pd.read_csv('../DATA/heart.csv')
df.head()

print(df['target'].unique())

# Exploratory Data Analysis and Visualization
df.info()
df.describe().transpose()

# Visualization 

sns.countplot(x='target',data=df)
print(df.columns)

# Running pairplot on everything will take a very long time to render!
sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')

#plotting heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)

#creating x and y data
X = df.drop('target',axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Logistic Regression Model
# we can use both logicticragressioncv or else combine logisticregression and grigsearchCv 
# help(LogisticRegressionCV)

log_model = LogisticRegressionCV()
log_model.fit(scaled_X_train,y_train)

# viewing the best params on which the data is fitted
log_model.C_
log_model.get_params()


# Coeffecients

coefs = pd.Series(index=X.columns,data=log_model.coef_[0])
coefs = coefs.sort_values()

plt.figure(figsize=(10,6))
sns.barplot(x=coefs.index,y=coefs.values);

#plotting confusion_matrix,classification_report,etc

y_pred = log_model.predict(scaled_X_test)

confusion_matrix(y_test,y_pred)
plot_confusion_matrix(log_model,scaled_X_test,y_test)
print(classification_report(y_test,y_pred))

# Performance Curves

plot_precision_recall_curve(log_model,scaled_X_test,y_test)

plot_roc_curve(log_model,scaled_X_test,y_test)

#predicting on some other sample data

patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]

X_test.iloc[-1]

y_test.iloc[-1]

log_model.predict(patient)

#predicting the probability values
log_model.predict_proba(patient)

#Bye....
#bye...
