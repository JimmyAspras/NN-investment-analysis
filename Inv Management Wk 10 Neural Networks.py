#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Please  construct  a  new  dataset  by  either  adding  two  independent  variables  or  removing  two independent
#variables  from  finalsample.dta  dataset.  If  you  choose  to  add  two  independent variables, you could add
#any two independent variables that you think help explain stock returns. If  you  choose  to  remove  two  
#independent  variables,  you  could  remove  any  two  independent variables that already exist in the 
#finalsample.dta dataset.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
plt.rcParams['figure.figsize'] = [20, 15]


# In[2]:


nndata=pd.read_stata('/Users/jimmyaspras/Downloads/finalsample.dta')
nndata.columns


# In[3]:


nndata.sort_values(by=['datadate'], inplace=True)
nndata1=nndata[nndata['lagPrice2']>=5]#remove penny stocks
nndata1['Year']=nndata1['datadate'].dt.year
nndata1['Month']=nndata1['datadate'].dt.month
#set gvkey and datadate as the index
nndata1=nndata1.set_index(['gvkey','datadate'])
nndata1.head()


# In[4]:


#Split  your  new  dataset  into  training  and  testing  samples.  Testing  sample  should  include  data with
#year>=2016. 
#
#Drop dvpspq and atq from the train/test data
#
train=nndata1[nndata1['Year']<2016]
X_train=train[['sic_2', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2']]


# In[5]:


#Set return as the dependent training variable
Y_train=train[['ret']]


# In[6]:


#Set testing independent variables
test=nndata1[nndata1['Year']>=2016]
X_test=test[['sic_2', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2']]


# In[7]:


#Set return as the dependent testing variable
Y_test=test[['ret']]


# In[8]:


#Calculate avg monthly risk free return
rf1=pd.read_excel("/Users/jimmyaspras/Downloads/Treasury bill.xlsx")
rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


# In[9]:


#Import benchmark index return
indexret1=pd.read_stata("/Users/jimmyaspras/Downloads/Index return.dta")
#Import factors data
Factor=pd.read_excel("/Users/jimmyaspras/Downloads/Factors.xlsx")


# In[10]:


#Build  a  neural  network  with  one  hidden  layer  and  20  neurons  in  the  hidden  layer.  Set  batch 
#size=10,000.  Use  GridSearchCV  to  search  for  the  best  value  of  epochs  among  [10,  20,  30,  40]. Use  
#the  best value  of  epochs  found  in  the  search  to  train  this  neural  network  using  your  new training 
#sample. Use the trained neural network to predict returns based on your new testing sample. Report the average 
#return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. 
#Also, report the Sharpe ratio of the portfolio. 
def shallownetwork():
    model=Sequential()
    model.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model


# In[11]:


tsplit=TimeSeriesSplit(n_splits=5,test_size=50000,gap=5000)
model1=KerasRegressor(build_fn=shallownetwork)
param_candidate={'epochs': [10,20,30,40]}
grid=GridSearchCV(estimator=model1,param_grid=param_candidate,n_jobs=-1,cv=tsplit,scoring='neg_mean_squared_error')
grid.fit(X_train,Y_train,batch_size=10000,verbose=0)
grid.cv_results_


# In[12]:


grid.best_params_


# In[13]:


nnretrain=shallownetwork()
#**grid.best_params_ feeds epochs directly into the model fitting
nnretrain.fit(X_train,Y_train,**grid.best_params_,batch_size=10000,verbose=1)


# In[14]:


Y_predict=pd.DataFrame(nnretrain.predict(X_test),columns=['Y_predict'])


# In[15]:


Y_test1=pd.DataFrame(Y_test).reset_index()


# In[16]:


Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[ ]:


#Return of 1.49% over market


# In[17]:


Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR


# In[18]:


#Build a deep neural network with more than 2 hidden layers. Feel free to pick the number of hidden layers. Use 
#RandomizedSearchCV to search for the best values of epochs, batch size, and the  number  of  neurons  in  each  
#hidden  layer.  Use  the  best  values  of  epochs,  batch  size,  and  the number of neurons in each hidden layer
#found in the search to train the deep neural network using your new training sample. Use the trained deep neural 
#network to predict returns based on your new testing sample. Report the average return of the portfolio that 
#consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio 
#of the portfolio.  


# In[19]:


def deepnetwork(no_neuron1,no_neuron2,no_neuron3,no_neuron4,no_neuron5,no_neuron6,no_neuron7,no_neuron8,no_neuron9,no_neuron10):
    model=Sequential()
    model.add(Dense(no_neuron1, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron2, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron3, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron4, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron5, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron6, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron7, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron9, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(no_neuron10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


# In[20]:


deepnetworktsplit=TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)


# In[21]:


param_dist = {'no_neuron1': randint(10,100),
              'no_neuron2': randint(10,100),
              'no_neuron3': randint(10,100),
              'no_neuron4': randint(10,100),
              'no_neuron5': randint(10,100),
              'no_neuron6': randint(10,100),
              'no_neuron7': randint(10,100),
              'no_neuron8': randint(10,100),
              'no_neuron9': randint(10,100),
              'no_neuron10': randint(10,100),
              'epochs': randint(10,50),
              'batch_size': randint(10000,50000)}


# In[22]:


model2 = KerasRegressor(build_fn=deepnetwork)
rgrid = RandomizedSearchCV(estimator=model2, param_distributions=param_dist, n_iter=5, cv=deepnetworktsplit,
                           scoring='neg_mean_squared_error', n_jobs=-1)
rgrid.fit(X_train,Y_train,verbose=0)
rgrid.cv_results_


# In[23]:


rgrid.best_params_


# In[24]:


#rgrid.best_params_['no_neuronX'] feeds neuron output from best params into model
def deepnetwork1():
    model = Sequential()
    model.add(Dense(rgrid.best_params_['no_neuron1'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron2'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron3'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron4'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron5'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron6'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron7'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron8'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron9'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(rgrid.best_params_['no_neuron10'], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model


# In[25]:


deep_n=deepnetwork1()
#best params fed directky into model fit
deep_n.fit(X_train,Y_train,epochs=rgrid.best_params_['epochs'],batch_size=rgrid.best_params_['batch_size'],verbose=1)


# In[26]:


Y_predict2=pd.DataFrame(deep_n.predict(X_test),columns=['Y_predict'])


# In[27]:


Y_test2=pd.DataFrame(Y_test).reset_index()


# In[28]:


Comb1=pd.merge(Y_test2, Y_predict2, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
stock_long2['datadate'].value_counts()
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']
stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']
stock_long5=sm.add_constant(stock_long5)
sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()


# In[ ]:


#1.1% over market


# In[29]:


Ret_rfdeepnetwork=stock_long5[['ret_rf']]
SRdeepnetwork=(Ret_rfdeepnetwork.mean()/Ret_rfdeepnetwork.std())*np.sqrt(12)
SRdeepnetwork

