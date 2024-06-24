import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


#This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

#load dataset
filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(filepath, header=0)
df.head()

#shape
df.dtypes
#summary
df.describe()

#data wrangling
df.drop(['id','Unnamed: 0'],axis=1,inplace=True)
df.head()

#missing values
print('Number of NaN values for the column bedrooms:',df['bedrooms'].isnull().sum())
print('Number of NaN values for the column bathrooms:',df['bathrooms'].isnull().sum())


#We can replace the missing values of the column 'bedrooms' and 'bathrooms' with the mean of the colunm
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean,inplace=True)

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean,inplace=True)

print('Number of NaN values for the column bedrooms:',df['bedrooms'].isnull().sum())
print('Number of NaN values for the column bathrooms:',df['bathrooms'].isnull().sum())


#Exploratory Data Analysis
house_unique_floorvalue = df['floors'].value_counts()
house_unique_floorvaluedf = house_unique_floorvalue.to_frame()

#Use the function boxplot in the seaborn library  to  
#determine whether houses with a waterfront view or without a waterfront view have more price outliers.
sns.boxplot(x='waterfront',y='price',data=df)
plt.title=('Distribution of Prices for Houses with and without Waterfront View')
plt.xlabel=('waterfront view')
ylabel=('price')
plt.xticks([0, 1], ['No Waterfront', 'Waterfront'])  # Customize x-axis ticks
plt.grid(True)
plt.show()


#Use the function regplot  in the seaborn library  to  
#determine if the feature sqft_above is negatively or positively correlated with price.
sns.regplot(x='sqft_above',y='price',data=df)
plt.title('Correlation between sqft_above and Price')
plt.xlabel('Sqft Above')
plt.ylabel('Price')
plt.grid(True)
plt.show()
df1=df.drop(['date'],axis=1)
df1.corr()['price'].sort_values()


#Model devolopment

#Fit a linear regression model using the  longitude feature'long' and  caculate the R^2.
x=df[['long']]
y=df['price']
lm=LinearRegression()
lm.fit(x,y)
lm.score(x,y)

#Fit a linear regression model to predict the 'price' using the feature 'sqft_living' then calculate the R^2
lm1=LinearRegression()
X=df[['sqft_living']]
Y=df['price']
lm1.fit(X,Y)
y_predict=lm1.predict(X)
lm1.score(X,Y)


#Fit a linear regression model to predict the 'price' using the list of features.
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] 
X=df[features]
Y=df['price']
lm2=LinearRegression()
lm2.fit(X,Y)
Y_predict=lm2.predict(X)
lm2.score(X,Y_predict)


#Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list features and calculate the R^2.
#model evaluation andrefinement

steps=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(steps)
pipe.fit(X,Y)
pred=pipe.predict(X)
r2=r2_score(Y,pred)
print('R^2 score:',r2)

#splitting data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15, random_state=1)
print('Number of test samples:',x_test.shape[0])
print('number of training samples:',x_train.shape[0])

# Ridge regression object using the training data, set the regularization parameter
ridg_e = Ridge(alpha=0.1)
ridg_e.fit(x_train, y_train)
y_pred = ridg_e.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("R^2 score:", r2)


ridge=Ridge(alpha=0.1)
ridge.fit(x_train,y_train)
y_pred=ridge.predict(x_test)
r2=r2_score(y_test,y_pred)
print('R^w score:', r2)


#Perform a second order polynomial transform on both the training data and testing data. 
poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
ridge=Ridge(alpha=0.1)
ridge.fit(x_train_poly,y_train)
y_pred=ridge.predict(x_test_poly)
r2=r2_score(y_test,y_pred)
print('R^2 score:', r2)