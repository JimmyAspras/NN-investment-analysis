# NN-investment-analysis

## Introuction

Can neural network models be used to beat the market?

Data for this project was chosen and downloaded from Wharton Research Data Services: https://wrds-www.wharton.upenn.edu/. This was done as part of a course taken in Summer 2021 complete with prompts and analysis. Credit must be given to my professor, Dr. Wei Jiao, for much code and instruction included here.

This project follows a series of prompts to determine if a neural network model can be refined to beat market return. The first half of the project builds neural network models, and the second half explores parameter tuning to enhance them.

## Preparing the Data

**Please construct a new dataset by either adding two independent variables or removing two independent variables from finalsample.dta dataset. If you choose to add two independent variables, you could add any two independent variables that you think help explain stock returns. If you choose to remove two independent variables, you could remove any two independent variables that already exist in the finalsample.dta dataset.**

### Libraries
```python
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
```

### Importing and Working with the Data

```python
nndata=pd.read_stata('/Users/jimmyaspras/Downloads/finalsample.dta')
nndata.columns
```

```python
Index(['gvkey', 'datadate', 'sic_2', 'lagdate', 'lagRet2', 'lagVOL2',
       'lagPrice2', 'lagMV2', 'lagShareturnover2', 'lagRet2_sic', 'lagRet12',
       'lagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
       'lagRet12_max', 'lagRet12_sic', 'lagdatadate', 'atq', 'ceqq', 'cheq',
       'dlttq', 'epspiq', 'saleq', 'dvpspq', 'sp500_ret_d', 'nasdaq_ret_d',
       'r2000_ret_d', 'dollar_ret_d', 'VIX', 'yield_3m', 'yield_10y',
       'gdp_growth', 'Bull_ave', 'Bull_Bear', 'ret', 'debt', 'cash', 'sale',
       'BM', 'PE', 'div_p', 'loglagPrice2', 'loglagVOL12', 'loglagMV2',
       'logatq', 'loglagVOL2'],
      dtype='object')
```

We want to sort the data date and exclude penny stocks. We also create vectors for year and month for the analysis and set the stock key and datadate as the index.

```python
nndata.sort_values(by=['datadate'], inplace=True)
nndata1=nndata[nndata['lagPrice2']>=5]#remove penny stocks
nndata1['Year']=nndata1['datadate'].dt.year
nndata1['Month']=nndata1['datadate'].dt.month
#set gvkey and datadate as the index
nndata1=nndata1.set_index(['gvkey','datadate'])
nndata1.head()
```

**Split your new dataset into training and testing samples. Testing sample should include data with year>=2016. I chose to drop dvpspq (dividends) and atq (total assets) from the train/test data.**

```python
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
```

Set return as the dependent training variable

```python
Y_train=train[['ret']]
```

Set testing independent variables

```python
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
```

Set return as the dependent testing variable

```python
Y_test=test[['ret']]
```

Calculate avg monthly risk free return

```python
rf1=pd.read_excel("/Users/jimmyaspras/Downloads/Treasury bill.xlsx")
rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()
```

Import benchmark index return and factors data

```python
indexret1=pd.read_stata("/Users/jimmyaspras/Downloads/Index return.dta")
Factor=pd.read_excel("/Users/jimmyaspras/Downloads/Factors.xlsx")
```

**Build a neural network with one hidden layer and 20 neurons in the hidden layer. Set batch size=10,000. Use GridSearchCV to search for the best value of  epochs among [10, 20, 30, 40]. Use the best value of epochs found in the search to train this neural network using your new training sample. Use the trained neural network to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio. 

Create the model
```python
def shallownetwork():
    model=Sequential()
    model.add(Dense(20,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    return model
```

Determine best number of epochs to use in the model
```python
tsplit=TimeSeriesSplit(n_splits=5,test_size=50000,gap=5000)
model1=KerasRegressor(build_fn=shallownetwork)
param_candidate={'epochs': [10,20,30,40]}
grid=GridSearchCV(estimator=model1,param_grid=param_candidate,n_jobs=-1,cv=tsplit,scoring='neg_mean_squared_error')
grid.fit(X_train,Y_train,batch_size=10000,verbose=0)
grid.cv_results_
```

Display best params
```python
grid.best_params_
```

```python
{'epochs': 30}
```

The ideal number of epochs to use in our model out of 10, 20, 30, and 40 is 30.

Train the model with ideal epochs, 30

```python
nnretrain=shallownetwork()
#**grid.best_params_ feeds epochs directly into the model fitting
nnretrain.fit(X_train,Y_train,**grid.best_params_,batch_size=10000,verbose=1)
```

```python
Y_predict=pd.DataFrame(nnretrain.predict(X_test),columns=['Y_predict'])
```

```python
Y_test1=pd.DataFrame(Y_test).reset_index()
```

Predict and rank returns
```python
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
```

Model summary

<img width="405" alt="image" src="https://user-images.githubusercontent.com/72087263/188293332-dc8a4040-7ce6-41ee-a14d-aebb81c513b9.png">

**The model produces a return of 1.49% above the market.**

Sharpe Ratio

```python
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
```

```python
ret_rf    0.755798
```

**Build a deep neural network with more than 2 hidden layers. Feel free to pick the number of hidden layers. Use RandomizedSearchCV to search for the best values of epochs, batch size, and the number of neurons in each hidden layer. Use the best values of epochs, batch size, and the number of neurons in each hidden layer found in the search to train the deep neural network using your new training sample. Use the trained deep neural network to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.**

```python
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
```

```python
deepnetworktsplit=TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)
```

```python
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
```

```python
model2 = KerasRegressor(build_fn=deepnetwork)
rgrid = RandomizedSearchCV(estimator=model2, param_distributions=param_dist, n_iter=5, cv=deepnetworktsplit,
                           scoring='neg_mean_squared_error', n_jobs=-1)
rgrid.fit(X_train,Y_train,verbose=0)
rgrid.cv_results_
```

Display best parameters

```python
rgrid.best_params_
```

```python
{'batch_size': 38088,
 'epochs': 32,
 'no_neuron1': 10,
 'no_neuron10': 31,
 'no_neuron2': 19,
 'no_neuron3': 62,
 'no_neuron4': 34,
 'no_neuron5': 69,
 'no_neuron6': 98,
 'no_neuron7': 82,
 'no_neuron8': 14,
 'no_neuron9': 82}
 ```
 
```python
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
```

```python
deep_n=deepnetwork1()
#best params fed directky into model fit
deep_n.fit(X_train,Y_train,epochs=rgrid.best_params_['epochs'],batch_size=rgrid.best_params_['batch_size'],verbose=1)
```

```python
Y_predict2=pd.DataFrame(deep_n.predict(X_test),columns=['Y_predict'])
```

```python
Y_test2=pd.DataFrame(Y_test).reset_index()
```

```python
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
```

Model summary

<img width="401" alt="image" src="https://user-images.githubusercontent.com/72087263/188294327-62535ce6-831e-4efb-8261-9e1f49fc844d.png">

**The model produces a return of 1.1% over the market.**

Sharpe Ratio
```python
Ret_rfdeepnetwork=stock_long5[['ret_rf']]
SRdeepnetwork=(Ret_rfdeepnetwork.mean()/Ret_rfdeepnetwork.std())*np.sqrt(12)
SRdeepnetwork
```

```python
ret_rf    0.88788
```

### Additional Libraries for Hypertuning

```python
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
```

**Build a neural network model. Give your own values for the number of hidden layers, epochs, batch size, the number of neurons in each hidden layer, and other hyperparameters. Run permutation_importance and report the feature importance graph (You could report either the original graph based on mean squared error-MSE or the scaled graph).

```python
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
```

Train the model

```python
model=KerasRegressor(build_fn=deepnetwork)
model.fit(X_train,Y_train,epochs=30,batch_size=40000,verbose=1)
```

Create variable for permutation_importance
```python
FIM=permutation_importance(model,X_train,Y_train,n_repeats=3,scoring='neg_mean_squared_error')
```

Create variables for mean, standard deviation, and score of FIM
```python
FIM_score_mean=pd.DataFrame(FIM.importances_mean,columns=['Feature Importance'])
FIM_score_std=pd.DataFrame(FIM.importances_std,columns=['Feature Importance_std'])
FIM_score=pd.merge(FIM_score_mean,FIM_score_std,left_index=True,right_index=True)
FIM_score['Feature']=X_test.columns.tolist()
FIM_score.sort_values(by=['Feature Importance'],inplace=True)
```

Plot feature importance

```python
FIM_score.plot(kind="barh",x='Feature',y='Feature Importance',title="Feature Importance",xerr='Feature Importance_std',fontsize=25,color='red')
```

![image](https://user-images.githubusercontent.com/72087263/188294589-fd21e79e-084e-4973-8a91-15a45c91e792.png)

**Use Random forest, Extra tree, HistGradientBoostingRegressor, and Neural network to do the model  ensemble. Give your own values for the hyperparameters in these four models. Use the average of the predictions of these four models as the final prediction on stock returns. Train this model ensemble using your new training sample. And use this model ensemble to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.

Build Random Forest model
```python
Randomforestensemble=RandomForestRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,
                                           max_samples=0.5,n_jobs=-1)
Randomforestensemble.fit(X_train,Y_train.values.ravel())
```

Build Extra Tree model
```python
ExTensemble= ExtraTreesRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,max_samples=0.5,n_jobs=-1)
ExTensemble.fit(X_train,Y_train.values.ravel())
```

Build Histgradient model
```python
HGBRensemble= HistGradientBoostingRegressor(max_iter=120,min_samples_leaf=120,early_stopping='False')     
HGBRensemble.fit(X_train,Y_train.values.ravel())
```

Use neural net to build ensemble
```python
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
```

Train the net
```python
Deepnetworkensemble=deepnetworkensemble()
Deepnetworkensemble.fit(X_train,Y_train,epochs=25,batch_size=40000,verbose=1)
```

Predictions
```python
Y_predict_Randomforestensemble=pd.DataFrame(Randomforestensemble.predict(X_test),columns=['Y_predict_Randomforestensemble'])
Y_predict_ExTensemble=pd.DataFrame(ExTensemble.predict(X_test),columns=['Y_predict_ExTensemble'])
Y_predict_HGBRensemble=pd.DataFrame(HGBRensemble.predict(X_test),columns=['Y_predict_HGBRensemble'])
Y_predict_Deepnetworkensemble=pd.DataFrame(Deepnetworkensemble.predict(X_test),columns=['Y_predict_Deepnetworkensemble'])
```

Prediction merge
```python
Y_predict1=pd.merge(Y_predict_HGBRensemble,Y_predict_Deepnetworkensemble,left_index=True,right_index=True)
Y_predict2=pd.merge(Y_predict1,Y_predict_ExTensemble,left_index=True,right_index=True)
Y_predict3=pd.merge(Y_predict2,Y_predict_Randomforestensemble,left_index=True,right_index=True)
```

Average predicted return
```python
Y_predict3['Y_predict']=Y_predict3.mean(1)
```

Merge predicted and actual, select 100 best performing, determine performance
```python
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
```

<img width="407" alt="image" src="https://user-images.githubusercontent.com/72087263/188294961-12eb5fec-a7ee-4c96-a69a-29d2cda217bb.png">

Sharpe ratio
```python
Ret_rf=stock_long5[['ret_rf']]
SRavgensemble=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SRavgensemble
```

```python
ret_rf    8.223031
```

**Use Random forest, Extra tree, HistGradientBoostingRegressor, and Neural network to do the model ensemble. Give your own values for the hyperparameters in the four models. Choose a linear machine learning model as the final model to combine the predictions of the four models. Train this model ensemble using your new training sample. And use this model ensemble to predict returns based on your new testing sample. Report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of the portfolio.**

Build the Random forest model
```python
Randomforestensemble1=RandomForestRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,
                                           max_samples=0.5,n_jobs=-1)
Randomforestensemble1.fit(X_train,Y_train.values.ravel())
```


Build Extra tree model
```python
ExTensemble1= ExtraTreesRegressor(n_estimators=120,min_samples_leaf=120,bootstrap=True,max_samples=0.5,n_jobs=-1)
ExTensemble1.fit(X_train,Y_train.values.ravel())
```

Build Histgradient model
```python
HGBRensemble1= HistGradientBoostingRegressor(max_iter=120,min_samples_leaf=120,early_stopping='False')     
HGBRensemble1.fit(X_train,Y_train.values.ravel())
```

Use neural net to build model ensemble
```python
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
```

```python
Deepnetworkensemble1=deepnetworkensemble1()
Deepnetworkensemble1.fit(X_train,Y_train,epochs=25,batch_size=40000,verbose=1)
```

Predictions
```python
Y_train_Randomforestensemble1=pd.DataFrame(Randomforestensemble1.predict(X_test),columns=['Y_predict_Randomforestensemble1'])
Y_train_ExTensemble1=pd.DataFrame(ExTensemble1.predict(X_test),columns=['Y_predict_ExTensemble1'])
Y_train_HGBRensemble1=pd.DataFrame(HGBRensemble1.predict(X_test),columns=['Y_predict_HGBRensemble1'])
Y_train_Deepnetworkensemble1=pd.DataFrame(Deepnetworkensemble1.predict(X_test),columns=['Y_predict_Deepnetworkensemble1'])
```

Prediction merge
```python
Y_train1=pd.merge(Y_train_HGBRensemble1,Y_train_Deepnetworkensemble1,left_index=True,right_index=True)
Y_train2=pd.merge(Y_train1,Y_train_ExTensemble1,left_index=True,right_index=True)
Y_train3=pd.merge(Y_train2,Y_train_Randomforestensemble1,left_index=True,right_index=True)
Y_train_new=pd.DataFrame(Y_train).reset_index()
Y_train4=pd.merge(Y_train3,Y_train_new,left_index=True,right_index=True)
```

```python
tsplit=TimeSeriesSplit(n_splits=10,test_size=10000, gap=5000)
```

```python
alpha_candidate=np.linspace(0.001,10,20)
finalmodel=RidgeCV(alphas=alpha_candidate,cv=tsplit)
```

```python
finalmodel.fit(Y_train4[['Y_predict_HGBRensemble1','Y_predict_Deepnetworkensemble1','Y_predict_ExTensemble1',
                          'Y_predict_Randomforestensemble1']],Y_train4['ret'])
```

```python
finalmodel.coef_
```

Output
```python
array([-0.00711704, -0.00213184, -0.02500874,  0.01057493])
```

Test sample predictions
```python
Y_predict_Randomforestensemble1=pd.DataFrame(Randomforestensemble1.predict(X_test), columns=['Y_predict_Randomforestensemble1'])
Y_predict_ExTensemble1=pd.DataFrame(ExTensemble1.predict(X_test),columns=['Y_predict_ExTensemble1'])
Y_predict_HGBRensemble1=pd.DataFrame(HGBRensemble1.predict(X_test),columns=['Y_predict_HGBRensemble1'])
Y_predict_Deepnetworkensemble1=pd.DataFrame(Deepnetworkensemble1.predict(X_test),columns=['Y_predict_Deepnetworkensemble1'])
```

Predictions merge
```python
Y_predict1=pd.merge(Y_predict_HGBRensemble1,Y_predict_Deepnetworkensemble1,left_index=True,right_index=True)
Y_predict2=pd.merge(Y_predict1,Y_predict_ExTensemble1,left_index=True,right_index=True)
Y_predict3=pd.merge(Y_predict2,Y_predict_Randomforestensemble1,left_index=True,right_index=True)

Final predictions
```python
Y_predict3['Y_predict']=finalmodel.predict(Y_predict3[['Y_predict_HGBRensemble1','Y_predict_Deepnetworkensemble1',
                                                       'Y_predict_ExTensemble1','Y_predict_Randomforestensemble1']])
```

Merge predicted and actual
```python
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
```

Model Summary
<img width="405" alt="image" src="https://user-images.githubusercontent.com/72087263/188295265-438c07aa-3c44-499c-a0e9-846048007de6.png">

Sharpe ratio
```python
Ret_rf=stock_long5[['ret_rf']]
SRlinensemble=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SRlinensemble
```

```python
ret_rf   -12.990851
```
