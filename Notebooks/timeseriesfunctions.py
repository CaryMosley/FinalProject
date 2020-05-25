def DickeyFullerTest(time_series, alpha = .05):
    '''This function takes in a time series and then outputs and formats the results of
    a Dickey-Fuller test You can also optionally send in a specific p value.
    '''
    #initialize test
    DFTest = adfuller(time_series)
    
    #results
    results = pd.Series(DFTest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    if results[1] > alpha:
        print('We fail to reject the null hypothesis that there is a unit root as our p-value of'
              , round(results[1],3), 'is greater than our alpha of', alpha,'\n')
    else:
        print('Our p-value of', round(results[1],3), 'is less than our alpha of', alpha,
              'so we reject the null hypothesis that there is a unit root. The data is stationary.\n')
    print ('Dickey-Fuller test: \n')
    print(results)
    
def ARIMA_models(time_series,AR_terms, MA_terms):
    '''This function takes in a timeseries and a set of AR and MA terms to try. The time series
    that it takes in needs to be already stationary. The function then returns a data frame
    of the model results'''
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    
    #Create results dataframe
    results = pd.DataFrame(columns=['Order', 'AIC', 'BIC', 'Test RMSE'])
    
    #ignore orders that do not converge or cause other erros
    for p in AR_terms:
        for q in MA_terms:
            try:
                order = (p,0,q)
                
            #create the ARIMA model and forecast
                time_series_model = ARIMA(time_series_train,order=order)
                time_series_fitted = time_series_model.fit()
                
                forecast, error, confidence_interval = time_series_fitted.forecast(len(time_series_test),alpha=.05)
                test_rmse = np.sqrt(mean_squared_error(time_series_test.values,forecast))
                results = results.append({'Order': order, 'AIC' : round(time_series_fitted.aic,3),
                           'BIC': round(time_series_fitted.bic,3),'Test RMSE': round(test_rmse,3)},ignore_index=True)
            except:
                print('Order',order,'caused an error.')
                continue
    return results

def plot_arima(time_series, AR = 0, MA = 0):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    order = (AR,0,MA)
         
    #create model and forecasts
    time_series_model = ARIMA(endog = time_series_train, order=order)
    time_series_fit = time_series_model.fit()
               
    forecast, error, confidence_interval = time_series_fit.forecast(len(time_series_test), alpha=0.05)
    # Make as pandas series
    forecast_time_series = pd.Series(forecast, index=time_series_test.index)
    lower_bound = pd.Series(confidence_interval[:, 0], index=time_series_test.index)
    upper_bound = pd.Series(confidence_interval[:, 1], index=time_series_test.index)
   
    # Plot the data and forecast
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(time_series_train, label='Train Data')
    plt.plot(time_series_test, label='Actual Values')
    plt.plot(forecast_time_series, label='Forecasted Values')
    plt.fill_between(time_series_test.index, lower_bound, upper_bound, alpha=.25)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    
def arima_summary(time_series,AR=0,MA =0):
    """
    This function will return the summary of the model.
    """
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    order = (AR,0,MA)
    print(ARIMA(time_series_train, order=order).fit().summary())
    
def plot_arima_forecasts(time_series_diff,time_series, AR = 0, MA = 0):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series_diff[:train_end]
    time_series_test = time_series_diff[test_start:]
    values_train = time_series[:train_end]
    values_test = time_series[test_start:]
    order = (AR,0,MA)
         
    #create model and forecasts
    time_series_model = ARIMA(endog = time_series_train, order=order)
    time_series_fit = time_series_model.fit()
    
               
    forecast, error, confidence_interval = time_series_fit.forecast(len(time_series_test), alpha=0.5)
    # Make as pandas series
    # Return to original scale
    forecast_time_series = pd.Series(forecast.cumsum(), index=time_series_test.index)
    forecast_time_series=forecast_time_series+values_train.iloc[-1].values
    
    # Plot the data and forecast
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(values_train, label='Train Data')
    plt.plot(values_test, label='Actual Values')
    plt.plot(forecast_time_series, label='Forecasted Values')
 
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
   
    plt.show()

def ARIMAX_models_single(time_series,AR_terms, MA_terms,exogenous):
    '''This function takes in a timeseries and a set of AR and MA terms to try. The time series
    that it takes in needs to be already stationary. The function then returns a data frame
    of the model results'''
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    
    #Create results dataframe
    results = pd.DataFrame(columns=['Order', 'AIC', 'BIC', 'Test RMSE', 'Exogenous'])
    
    #ignore orders that do not converge or cause other erros
    for p in AR_terms:
        for q in MA_terms:
            for column in exogenous:
                try:
                    order = (p,0,q)
                
                    #create the ARIMA model and forecast
                    time_series_model = ARIMA(endog = time_series_train, exog = exog_train[[column]], order = order)
                    time_series_fitted = time_series_model.fit()
                    forecast, error, confidence_interval = time_series_fitted.forecast(len(time_series_test),exog=exog_test[[column]], alpha=.05)
                    test_rmse = np.sqrt(mean_squared_error(time_series_test.values,forecast))
                    results = results.append({'Order': order, 'AIC' : round(time_series_fitted.aic,3),
                           'BIC': round(time_series_fitted.bic,3),'Test RMSE': round(test_rmse,3),
                                             'Exogenous': column},ignore_index=True)
                except:
                    print('Order',order,'caused an error.')
                    continue
           
    return results

def plot_ARIMAX(time_series, AR, MA, exogenous):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    order = (AR,0,MA)
         
    #create model and forecasts
    time_series_model = ARIMA(endog = time_series_train, exog=exog_train, order=order)
    time_series_fit = time_series_model.fit()
               
    forecast, error, confidence_interval = time_series_fit.forecast(len(time_series_test),exog=exog_test, alpha=0.05)
    # Make as pandas series
    forecast_time_series = pd.Series(forecast, index=time_series_test.index)
    lower_bound = pd.Series(confidence_interval[:, 0], index=time_series_test.index)
    upper_bound = pd.Series(confidence_interval[:, 1], index=time_series_test.index)
   
    # Plot the data and forecast
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(time_series_train, label='Train Data')
    plt.plot(time_series_test, label='Actual Values')
    plt.plot(forecast_time_series, label='Forecasted Values')
    plt.fill_between(time_series_test.index, lower_bound, upper_bound, 
                 color='k', alpha=.25)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)

    plt.show()
    
def ARIMAX_summary(time_series,AR,MA,exogenous):
    """
    This function will return the summary of the model.
    """
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    
    order = (AR,0,MA)
    print(ARIMA(time_series_train,exog=exog_train, order=order).fit().summary())
    
def plot_ARIMAX_forecasts(time_series_diff,time_series, AR, MA, exogenous):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series_diff[:train_end]
    time_series_test = time_series_diff[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    values_train = time_series[:train_end]
    values_test = time_series[test_start:]
    order = (AR,0,MA)
         
    #create model and forecasts
    time_series_model = ARIMA(endog = time_series_train, exog=exog_train, order=order)
    time_series_fit = time_series_model.fit()
               
    forecast, error, confidence_interval = time_series_fit.forecast(len(time_series_test),exog=exog_test, alpha=0.5)
    # Make as pandas series
    # Return to original scale
    forecast_time_series = pd.Series(forecast.cumsum(), index=time_series_test.index)
    forecast_time_series=forecast_time_series+values_train.iloc[-1].values
   
    # Plot the data and forecast
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(values_train, label='Train Data')
    plt.plot(values_test, label='Actual Values')
    plt.plot(forecast_time_series, label='Forecasted Values')
    
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()    
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """This function came from Jason Brownlee of Machine Learning Mastery.com
    Frame a time series as a supervised learning dataset. Arguments:
    mdata: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.Returns:
    Pandas DataFrame of series framed for supervised learning."""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg    


def VAR_models_single(time_series,AR_terms,exogenous):
    '''This function takes in multiple timeseries and a set of AR terms to try. It also takes in exogenous variables. The time series
    that it takes in needs to be already stationary. The function then returns a data frame using one exogenous variable at a time
    of the model results'''
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    
    #Create results dataframe
    results = pd.DataFrame(columns=['Lags','AIC', 'BIC', 'Test RMSE', 'Exogenous'])
    
    #ignore orders that do not converge or cause other erros
    for column in exogenous:
        for lag in AR_terms:
      
    #create the VAR model and forecast
            time_series_model = VAR(endog = time_series_train, exog = exog_train[[column]])
            time_series_fitted = time_series_model.fit(lag)
        
            forecasts = time_series_fitted.forecast(y= time_series_fitted.y, steps = len(time_series_test),exog_future=exog_test[[column]])
        
        
            test_rmse = np.sqrt(mean_squared_error(time_series_test.values,forecasts))
            results = results.append({'Lags': lag, 'AIC' : round(time_series_fitted.aic,3),
                           'BIC': round(time_series_fitted.bic,3),'Test RMSE': round(test_rmse,3),
                                             'Exogenous': column},ignore_index=True)
           
    return results

def VAR_models_combined(time_series,AR_terms,exogenous):
    '''This function takes in multiple timeseries and a set of AR terms to try. It also takes in exogenous variables. The time series
    that it takes in needs to be already stationary. The function then returns a data frame using all the exogenous variables at a time
    of the model results'''
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    
    #Create results dataframe
    results = pd.DataFrame(columns=['Lags','AIC', 'BIC', 'Test RMSE'])
    


    for lag in AR_terms:
      
    #create the VAR model and forecast
        time_series_model = VAR(endog = time_series_train, exog = exog_train)
        time_series_fitted = time_series_model.fit(lag)
        
        forecasts = time_series_fitted.forecast(y= time_series_fitted.y, steps = len(time_series_test),exog_future=exog_test)
        
        
        
        test_rmse = np.sqrt(mean_squared_error(time_series_test.values,forecasts))
        results = results.append({'Lags': lag, 'AIC' : round(time_series_fitted.aic,3),
                        'BIC': round(time_series_fitted.bic,3),'Test RMSE': round(test_rmse,3)
                        },ignore_index=True)
           
    return results

def VARMAX_models(time_series,AR_terms, MA_terms,exogenous):
    '''This function takes in multiple timeseries and a set of AR and MA terms to try. It also takes in exogenous variables. The time series
    that it takes in needs to be already stationary. The function then returns a data frame using all the exogenous variables at a time
    of the model results'''
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    
    #Create results dataframe
    results = pd.DataFrame(columns=['Order','AIC', 'BIC', 'Test RMSE'])
    


    for lag in AR_terms:
        for q in MA_terms:
            order = (lag,q)
    #create the VAR model and forecast
            time_series_model = sm.tsa.VARMAX(endog = time_series_train,order=order, exog = exog_train)
            time_series_fitted = time_series_model.fit(maxiter=1000)
        
            forecasts = time_series_fitted.forecast(steps = len(time_series_test),exog=exog_test)
           
        
            test_rmse = np.sqrt(mean_squared_error(time_series_test.values,forecasts))
            results = results.append({'Order': order, 'AIC' : round(time_series_fitted.aic,3),
                            'BIC': round(time_series_fitted.bic,3),'Test RMSE': round(test_rmse,3)
                            },ignore_index=True)
           
    return results

def johansen(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    results = coint_johansen(df,-1,3)
    d = {'0.90':0, '0.95':1, '0.99':2}
    test_stats = results.lr1
    cvts = results.cvt[:, d[str(1-alpha)]]


    # Summary
    print('Name   |  Test Stat > C(95%)    |   Reject Null  \n', '--'*20)
    for col, test_stat, cvt in zip(df.columns, test_stats, cvts):
        print(col, '| ', round(test_stat,2), ">", round(cvt, 2), ' |  ' , test_stat > cvt)

def plot_VAR(time_series, lag, exogenous):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    
         
    #create model and forecasts
    time_series_model = VAR(endog = time_series_train, exog=exog_train)
    time_series_fit = time_series_model.fit(lag)
               
    forecasts = time_series_fit.forecast(y= time_series_fit.y, steps = len(time_series_test),exog_future=exog_test)
    # Make as pandas series
    forecast_time_series = pd.DataFrame(forecasts, index=time_series_test.index)

   
    # Plot the data and forecast
    fig, (ax1, ax2) = plt.subplots(2,figsize=(20,20))

    
    ax1.plot(time_series_train.iloc[:,0], label='Train Data')
    ax1.plot(time_series_test.iloc[:,0], label='Actual Values')
    ax1.plot(forecast_time_series.iloc[:,0], label='Forecasted Values')
  
    ax1.set_title('Forecast vs Actuals')
    ax1.legend(loc='upper left', fontsize=8)
    
    
    ax2.plot(time_series_train.iloc[:,1], label='Train Data')
    ax2.plot(time_series_test.iloc[:,1], label='Actual Values')
    ax2.plot(forecast_time_series.iloc[:,1], label='Forecasted Values')
  
    ax2.set_title('Forecast vs Actuals')
    ax2.legend(loc='upper left', fontsize=8)
    plt.show()
    
def VAR_summary(time_series,lag,exogenous):
    """
    This function will return the summary of the model.
    """
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    

    print(VAR(time_series_train,exog=exog_train).fit(lag).summary())
    
    
def plot_VAR_forecasts(time_series_diff,time_series, lag, exogenous,series=0):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series_diff[:train_end]
    time_series_test = time_series_diff[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    values_train = time_series[:train_end]
    values_test = time_series[test_start:]
    
         
    #create model and forecasts
    time_series_model = VAR(endog = time_series_train, exog=exog_train)
    time_series_fit = time_series_model.fit(lag)
               
    forecasts = time_series_fit.forecast(y= time_series_fit.y, steps = len(time_series_test),exog_future=exog_test)    
    forecast_time_series = pd.DataFrame(forecasts, index=time_series_test.index)

    forecast_time_series = forecast_time_series.cumsum()
    forecast_time_series.iloc[:,0]=forecast_time_series.iloc[:,0]+values_train.iloc[:,0][-1]
    forecast_time_series.iloc[:,1]=forecast_time_series.iloc[:,1]+values_train.iloc[:,1][-1]

    # Plot the data and forecast
    fig, ax = plt.subplots(1,figsize=(20,10))

    
    ax.plot(values_train.iloc[:,series], label='Train Data')
    ax.plot(values_test.iloc[:,series], label='Actual Values')
    ax.plot(forecast_time_series.iloc[:,series], label='Forecasted Values')
  
    ax.set_title('Forecast vs Actuals')
    ax.legend(loc='upper left', fontsize=8)
    
 
    plt.show()      
    
    
    
def plot_VARMAX(time_series, AR_term, MA_term, exogenous):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    p = AR_term
    q = MA_term
    order = (p,q)
         
    #create model and forecasts
    time_series_model = sm.tsa.VARMAX(endog = time_series_train,order=order, exog = exog_train)
    time_series_fitted = time_series_model.fit(maxiter=1000)
               
    forecasts = time_series_fitted.forecast(steps = len(time_series_test),exog=exog_test)
    # Make as pandas series
    forecast_time_series = pd.DataFrame(forecasts, index=time_series_test.index)
    
   
    # Plot the data and forecast
    fig, (ax1, ax2) = plt.subplots(2,figsize=(20,20))

    
    ax1.plot(time_series_train.iloc[:,0], label='Train Data')
    ax1.plot(time_series_test.iloc[:,0], label='Actual Values')
    ax1.plot(forecast_time_series.iloc[:,0], label='Forecasted Values')
  
    ax1.set_title('Forecast vs Actuals')
    ax1.legend(loc='upper left', fontsize=8)
    
    
    ax2.plot(time_series_train.iloc[:,1], label='Train Data')
    ax2.plot(time_series_test.iloc[:,1], label='Actual Values')
    ax2.plot(forecast_time_series.iloc[:,1], label='Forecasted Values')
  
    ax2.set_title('Forecast vs Actuals')
    ax2.legend(loc='upper left', fontsize=8)
    plt.show()  
    return

def VARMAX_summary(time_series,AR_term, MA_term, exogenous):
    """
    This function will return the summary of the model.
    """
    time_series_train = time_series[:train_end]
    time_series_test = time_series[test_start:]
    
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    p = AR_term
    q = MA_term
    order = (p,q)

    print(sm.tsa.VARMAX(time_series_train,exog=exog_train,order = order).fit().summary())

def plot_VARMAX_forecasts(time_series_diff,time_series, AR_term,MA_term, exogenous,series=0):
    """
    This function will graph the time series including the forecast data.
    """
    #I'm going to break the time series into a train and test set using the earlier data as the train
    time_series_train = time_series_diff[:train_end]
    time_series_test = time_series_diff[test_start:]
    exog_train = exogenous[:train_end]
    exog_test = exogenous[test_start:]
    values_train = time_series[:train_end]
    values_test = time_series[test_start:]
    p = AR_term
    q = MA_term
    order = (p,q)
    
         
    #create model and forecasts
    time_series_model = sm.tsa.VARMAX(endog = time_series_train,order=order, exog = exog_train)
    time_series_fitted = time_series_model.fit(maxiter=1000)
               
    forecasts = time_series_fitted.forecast(steps = len(time_series_test),exog=exog_test) 
    forecast_time_series = pd.DataFrame(forecasts, index=time_series_test.index)

    forecast_time_series = forecast_time_series.cumsum()
    forecast_time_series.iloc[:,0]=forecast_time_series.iloc[:,0]+values_train.iloc[:,0][-1]
    forecast_time_series.iloc[:,1]=forecast_time_series.iloc[:,1]+values_train.iloc[:,1][-1]

    # Plot the data and forecast
    fig, ax = plt.subplots(1,figsize=(20,10))

    
    ax.plot(values_train.iloc[:,series], label='Train Data')
    ax.plot(values_test.iloc[:,series], label='Actual Values')
    ax.plot(forecast_time_series.iloc[:,series], label='Forecasted Values')
  
    ax.set_title('Forecast vs Actuals- SPY Prices')
    ax.legend(loc='upper left', fontsize=8)
    
    
  
    plt.show()
def LSTM_forecast(df,train_length ):
    non_predict = list(range(-1,-13,-1))
    #load values
    values = LSTM_model.values

    #Scale featuers
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    #specify number of lags
    lags = 1
    n_features = 13

    #convert to supervised
    supervised = series_to_supervised(scaled,lags,1)

    #drop columns we're not predicting
    supervised.drop(supervised.columns[non_predict], axis=1, inplace=True)
    values = supervised.values
    train_length = len(spy_diff['SPY Differenced'][:train_end])
    #split into train and test set
    train = values[:train_length, :]
    test = values[train_length:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))             
    # design network
    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    lstm_model = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y),     verbose=0, shuffle=False)
    
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], lags*n_features))
    
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]