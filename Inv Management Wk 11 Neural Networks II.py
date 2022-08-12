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
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
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


#Build a neural network model. Give your own values for the number of hidden layers, epochs, batch  size,  the  
#number  of  neurons  in  each  hidden  layer,  and  other  hyperparameters.  Run permutation_importance  and  
#report  the  feature  importance  graph  (You  could  report  either  the original graph based on mean squared 
#error-MSE or the scaled graph).
def deepnetwork():
    model = Sequential()
    model.add(Dense(15, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(63, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(52, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(27, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(86, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(78, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model


# In[11]:


model=KerasRegressor(build_fn=deepnetwork)
model.fit(X_train,Y_train,epochs=30,batch_size=40000,verbose=1)


# In[12]:


FIM=permutation_importance(model,X_train,Y_train,n_repeats=3,scoring='neg_mean_squared_error')


# In[13]:


FIM_score_mean=pd.DataFrame(FIM.importances_mean,columns=['Feature Importance'])
FIM_score_std=pd.DataFrame(FIM.importances_std,columns=['Feature Importance_std'])
FIM_score=pd.merge(FIM_score_mean,FIM_score_std,left_index=True,right_index=True)
FIM_score['Feature']=X_test.columns.tolist()
FIM_score.sort_values(by=['Feature Importance'],inplace=True)


# In[14]:


FIM_score.plot(kind="barh",x='Feature',y='Feature Importance',title="Feature Importance",xerr='Feature Importance_std',fontsize=25,color='red')


# In[15]:


FIM_score['benchmark']=FIM_score['Feature Importance'].max()
FIM_score['Feature Importance%']=FIM_score['Feature Importance']/FIM_score['benchmark']


# In[16]:


FIM_score.plot(kind="barh",x='Feature',y='Feature Importance%',title="Feature Importance",fontsize=20,color='red')


# In[17]:


#Use Random forest, Extra tree, HistGradientBoostingRegressor, and Neural network to do the model  ensemble.  
#Give  your  own  values  for  the  hyperparameters  in  these  four  models.  Use  the average of the predictions 
#of these four models as the final prediction on stock returns. Train this model ensemble using your new training 
#sample. And use this model ensemble to predict returns based on your new testing sample. Report the average 
#return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. 
#Also, report the Sharpe ratio of the portfolio.  


# In[18]:


#Random forest
Randomforestensemble=RandomForestRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,
                                           max_samples=0.5,n_jobs=-1)
Randomforestensemble.fit(X_train,Y_train.values.ravel())


# In[19]:


#Extra tree
ExTensemble= ExtraTreesRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,max_samples=0.5,n_jobs=-1)
ExTensemble.fit(X_train,Y_train.values.ravel())


# In[20]:


#Histgradient
HGBRensemble= HistGradientBoostingRegressor(max_iter=120,min_samples_leaf=120,early_stopping='False')     
HGBRensemble.fit(X_train,Y_train.values.ravel())


# In[21]:


#Neural net
def deepnetworkensemble():
    model = Sequential()
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model


# In[22]:


Deepnetworkensemble=deepnetworkensemble()
Deepnetworkensemble.fit(X_train,Y_train,epochs=25,batch_size=40000,verbose=1)


# In[23]:


#Predictions
Y_predict_Randomforestensemble=pd.DataFrame(Randomforestensemble.predict(X_test),columns=['Y_predict_Randomforestensemble'])
Y_predict_ExTensemble=pd.DataFrame(ExTensemble.predict(X_test),columns=['Y_predict_ExTensemble'])
Y_predict_HGBRensemble=pd.DataFrame(HGBRensemble.predict(X_test),columns=['Y_predict_HGBRensemble'])
Y_predict_Deepnetworkensemble=pd.DataFrame(Deepnetworkensemble.predict(X_test),columns=['Y_predict_Deepnetworkensemble'])


# In[24]:


#Prediction merge
Y_predict1=pd.merge(Y_predict_HGBRensemble,Y_predict_Deepnetworkensemble,left_index=True,right_index=True)
Y_predict2=pd.merge(Y_predict1,Y_predict_ExTensemble,left_index=True,right_index=True)
Y_predict3=pd.merge(Y_predict2,Y_predict_Randomforestensemble,left_index=True,right_index=True)


# In[25]:


#AVG predicted return
Y_predict3['Y_predict']=Y_predict3.mean(1)


# In[26]:


#Merge predicted and actual, select 100 best performing, determine performance
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1,Y_predict3,left_index=True,right_index=True,how='inner')
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


# In[27]:


#Sharpe ratio
Ret_rf=stock_long5[['ret_rf']]
SRavgensemble=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SRavgensemble


# In[28]:


#Use Random forest, Extra tree, HistGradientBoostingRegressor, and Neural network to do the model  ensemble.  
#Give  your  own  values  for  the  hyperparameters  in  the  four  models.  Choose  a linear machine learning 
#model as the final model to combine the predictions of the four models. Train this model ensemble using your new 
#training sample. And use this model ensemble to predict returns based on your new testing sample. Report the 
#average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each 
#year-month. Also, report the Sharpe ratio of the portfolio. 


# In[29]:


#Random forest
Randomforestensemble1=RandomForestRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,
                                           max_samples=0.5,n_jobs=-1)
Randomforestensemble1.fit(X_train,Y_train.values.ravel())


# In[30]:


#Extra tree
ExTensemble1= ExtraTreesRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,max_samples=0.5,n_jobs=-1)
ExTensemble1.fit(X_train,Y_train.values.ravel())


# In[31]:


#Histgradient
HGBRensemble1= HistGradientBoostingRegressor(max_iter=120,min_samples_leaf=120,early_stopping='False')     
HGBRensemble1.fit(X_train,Y_train.values.ravel())


# In[32]:


#Neural net
def deepnetworkensemble1():
    model = Sequential()
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(75, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model


# In[33]:


Deepnetworkensemble1=deepnetworkensemble1()
Deepnetworkensemble1.fit(X_train,Y_train,epochs=25,batch_size=40000,verbose=1)


# In[34]:


#Predictions
Y_train_Randomforestensemble1=pd.DataFrame(Randomforestensemble1.predict(X_test),columns=['Y_predict_Randomforestensemble1'])
Y_train_ExTensemble1=pd.DataFrame(ExTensemble1.predict(X_test),columns=['Y_predict_ExTensemble1'])
Y_train_HGBRensemble1=pd.DataFrame(HGBRensemble1.predict(X_test),columns=['Y_predict_HGBRensemble1'])
Y_train_Deepnetworkensemble1=pd.DataFrame(Deepnetworkensemble1.predict(X_test),columns=['Y_predict_Deepnetworkensemble1'])


# In[35]:


#Prediction merge
Y_train1=pd.merge(Y_train_HGBRensemble1,Y_train_Deepnetworkensemble1,left_index=True,right_index=True)
Y_train2=pd.merge(Y_train1,Y_train_ExTensemble1,left_index=True,right_index=True)
Y_train3=pd.merge(Y_train2,Y_train_Randomforestensemble1,left_index=True,right_index=True)
Y_train_new=pd.DataFrame(Y_train).reset_index()
Y_train4=pd.merge(Y_train3,Y_train_new,left_index=True,right_index=True)


# In[36]:


tsplit=TimeSeriesSplit(n_splits=10,test_size=10000, gap=5000)


# In[38]:


from sklearn.linear_model import RidgeCV


# In[39]:


alpha_candidate=np.linspace(0.001,10,20)
finalmodel=RidgeCV(alphas=alpha_candidate,cv=tsplit)


# In[40]:


finalmodel.fit(Y_train4[['Y_predict_HGBRensemble1','Y_predict_Deepnetworkensemble1','Y_predict_ExTensemble1',
                          'Y_predict_Randomforestensemble1']],Y_train4['ret'])


# In[43]:


finalmodel.coef_


# In[44]:


#Test sample predictions
Y_predict_Randomforestensemble1=pd.DataFrame(Randomforestensemble1.predict(X_test), columns=['Y_predict_Randomforestensemble1'])
Y_predict_ExTensemble1=pd.DataFrame(ExTensemble1.predict(X_test),columns=['Y_predict_ExTensemble1'])
Y_predict_HGBRensemble1=pd.DataFrame(HGBRensemble1.predict(X_test),columns=['Y_predict_HGBRensemble1'])
Y_predict_Deepnetworkensemble1=pd.DataFrame(Deepnetworkensemble1.predict(X_test),columns=['Y_predict_Deepnetworkensemble1'])


# In[45]:


#Predictions merge
Y_predict1=pd.merge(Y_predict_HGBRensemble1,Y_predict_Deepnetworkensemble1,left_index=True,right_index=True)
Y_predict2=pd.merge(Y_predict1,Y_predict_ExTensemble1,left_index=True,right_index=True)
Y_predict3=pd.merge(Y_predict2,Y_predict_Randomforestensemble1,left_index=True,right_index=True)


# In[46]:


#Final predictions
Y_predict3['Y_predict']=finalmodel.predict(Y_predict3[['Y_predict_HGBRensemble1','Y_predict_Deepnetworkensemble1',
                                                       'Y_predict_ExTensemble1','Y_predict_Randomforestensemble1']])


# In[47]:


#Merge predicted and actual
Y_test1=pd.DataFrame(Y_test).reset_index()
Comb1=pd.merge(Y_test1, Y_predict3, left_index=True,right_index=True,how='inner')
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


# In[48]:


#Sharpe ratio
Ret_rf=stock_long5[['ret_rf']]
SRlinensemble=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SRlinensemble

