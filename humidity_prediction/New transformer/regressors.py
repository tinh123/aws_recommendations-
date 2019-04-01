from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np


def linear_reg_model(x_train,y_train,verbose=True):
    if verbose:
        print('Linear Regression')
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def linear_svr(x_train,y_train,verbose=True):
    if verbose:
        print('Linear SVR')
    model = LinearSVR(random_state=0, tol=1e-5, max_iter=1000000)
    model.fit(x_train,y_train)
    return model

def adaboost_reg(x_train,y_train,verbose=True):
    if verbose:
        print('Adaboost Regression')
    rng = np.random.RandomState(1)
    model = AdaBoostRegressor(DecisionTreeRegressor(),random_state=rng,n_estimators=200)
    model.fit(x_train, y_train)
    return model

def gradient_boosting_reg(x_train, y_train,verbose=True):
    if verbose:
        print('Gradient Boosting Regression')
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(x_train, y_train)
    return model

def kernel_ridge(x_train, y_train,verbose=True):
    if verbose:
        print('Ridge Regression')
    rng = np.random.RandomState(0)
    model = KernelRidge(alpha=0.1)
    model.fit(x_train, y_train)
    return model

def linear_lasso(x_train, y_train,verbose=True):
    if verbose:
        print("Lasso Regression")
    model = linear_model.Lasso(alpha=0.1)
    model.fit(x_train,y_train)
    return model

def quantile_regression(x_train, y_train,verbose=True):
    if verbose:
        print("Quantile Regression")
    model = GradientBoostingRegressor(loss='quantile', alpha = 0.8, n_estimators= 150, max_depth=2)
    model.fit(x_train, y_train)
    return model


#%%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from desc_plot import plot_reg_result
from matplotlib import pyplot as plt

def show_reg_result(model, x_test, y_test,plot=True):   
    """
    show the result value and plot of regression models
    
    params
    ------
    model: sklearn model instance
    x_test: 
    """
    y_pred = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    mse = mean_squared_error(y_test, y_pred)
    
    if plot:
        print("MAE:",mae)
        print("RMSE:",rmse)
        print("MSE:",mse)    
        plot_reg_result(y_test, y_pred)
        
    return {"rmse":rmse,
            "mae":mae,
            "mse":mse}
#%%
from keras.utils import to_categorical
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense, Dropout\

def pipeline_reg(x_train,y_train,x_test,y_test,
                 models=['lin reg','lin svr','ada','grad','ridge','lasso','ann', 'quantile'],
                 plot=True):
    y_train = y_train.values.reshape(-1,)
    model_result_dict = {}
    if 'lin reg' in models:
        # Linear regression
        model = linear_reg_model(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['lin reg'] = (model,result)
        
    if 'lin svr' in models:
        # Linear svr
        model = linear_svr(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['lin svr'] = (model,result)
        
    if 'ada' in models:
        # Adaboost regressor
        model = adaboost_reg(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['ada'] = (model,result)
        
    if 'grad' in models:
        # Gradient boosting regression  
        model = gradient_boosting_reg(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['grad'] = (model,result)
        
    if 'ridge' in models:
        # Kernel ridge
        model = kernel_ridge(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['ridge'] = (model,result)
        
    if 'lasso' in models:
        # Linear_lasso
        model = linear_lasso(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['lasso'] = (model,result)
    if 'quantile' in models:
        # Quantile regression
        model = quantile_regression(x_train,y_train,verbose=plot)
        result = show_reg_result(model,x_test,y_test,plot)
        model_result_dict['quantile'] = (model,result)
    if 'ann' in models:
        # Input layer
        model = Sequential()
        model.add(Dense(50, input_dim=x_train.shape[1], activation='softsign')) # 1st hidden layer
        model.add(Dense(20, activation='softsign')) # Hidden layer
        model.add(Dense(1, activation='sigmoid')) # Output layer

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='sgd')
        # Fit the model
        model.fit(x_train, y_train, epochs=500, batch_size=50,verbose=False)

        # evaluate the model
        scores = model.evaluate(x_test, y_test)
        print(scores)
        y_pred = model.predict(x_test).T
        plot_reg_result(y_test,y_pred)
        model_result_dict['ann'] = (model,result)
        
    return model_result_dict

