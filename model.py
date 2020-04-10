'''

This is a small demo which is created for testing flask application and deployment in Heroku 
Data : Taxi data for number of riders as dependent variable
Features : Priceperweek,Population,Monthlyincome,Averageparkingpermonth,Numberofweeklyriders

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV,KFold)
import warnings
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns 
import pickle
from tqdm import tqdm


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
plot = False


df = pd.read_csv('taxi.csv')


if plot==True:
	print(df.head())
	print(df.describe([0.25,0.50,0.75,0.90,0.99]).T)
	plt.figure(figsize=(5,5))
	sns.heatmap(df.corr(),annot = True,cmap='winter')
	sns.pairplot(df.corr())
	plt.show()

X  = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=42)

# model1 = LinearRegression()
# model1.fit(train_X,train_y)

model2 = Ridge()
folds = KFold(n_splits=5,shuffle=True,random_state=42)
cvs = cross_val_score(model2,train_X,train_y,scoring='r2',cv=folds,n_jobs=-1,verbose=True)
print(cvs.mean())

param ={'alpha':[0.001,0.01,0.1,0.2,0.5,0.9,1.0, 5.0, 10.0,20.0,30.0,40.0]}

model_cv = GridSearchCV(model2,param_grid = param,scoring='r2',n_jobs=-1,
	cv=folds,verbose=1,return_train_score=True)

model_cv.fit(train_X,train_y)
print(model_cv.best_params_)
print(model_cv.best_score_)
cv_result  = model_cv.cv_results_
cv_result = pd.DataFrame(cv_result)
cv_result = cv_result[['param_alpha','mean_train_score','mean_test_score']]
cv_result['param_alpha'] =  cv_result['param_alpha'].astype('float32')


if plot==True:
	print(cv_result.head())
	plt.figure(figsize=(6,6))
	plt.plot(cv_result.param_alpha,cv_result.mean_train_score)
	plt.plot(cv_result.param_alpha,cv_result.mean_test_score)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
model_final = Ridge(alpha=40.0)
model_final.fit(train_X,train_y)
print(model_final.score(train_X,train_y))
print(model_final.score(test_X,test_y))


pickle.dump(model_final,open('model.pkl','wb'))