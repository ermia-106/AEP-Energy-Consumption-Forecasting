# Import Libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from sklearn.metrics import mean_absolute_error , mean_squared_error
print('======= Libraries are Imported =======')

# Load Dataset
dataset = pd.read_csv(r'E:\Programming\GitHub Repository\Python\AEP-Energy-Consumption-Forecasting\AEP_hourly.csv')
print('\n======= AEP Dataset is Imported =======')

# Data Preprocessing
print('\n======= Preprocessing Data =======')
print(dataset.head())
print('Dimensions of Dataset :',dataset.shape)
print(dataset.describe())
dataset.info()
dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])
dataset = dataset.sort_values('Datetime')
dataset = dataset.set_index('Datetime')
print('Missing Values :',dataset.isnull().sum())
print('Number of Duplicated Indexes :',dataset.index.duplicated().sum())
print(dataset[dataset.index.duplicated(keep = False)])
dataset = dataset.groupby(dataset.index).mean()
print('Number of Duplicated Indexes :',dataset.index.duplicated().sum())
dataset = dataset.asfreq('h')
dataset = dataset.interpolate(method = 'time')
print('Dimensions of Dataset :',dataset.shape)
print(dataset.head())
print(dataset.describe())
dataset.info()

# Exploratory Data Analysis ( EDA )
print('\n======= Primary Data\'s Chart =======')
def dataset_plot (time,values) :
    plt.figure(figsize = (15,5))
    plt.plot(time,values,label = 'AEP')
    plt.title('AEP Hourly Energy Consumption Chart')
    plt.xlabel('Datetime')
    plt.ylabel('Energy (MW)')
    plt.legend()
    plt.grid(True)
    plt.show()
dataset_plot(dataset.index,dataset['AEP_MW'])

# Stationarity Test ( ADF )
print('\n======= Stationarity Test ( ADF ) =======')
def stationarity (series) :
    adfuller_result = adfuller(series)
    print(f'ADF Statistic : {adfuller_result[0]}')
    print(f'p-value : {adfuller_result[1]}')
    print(f'Number of Lags Used : {adfuller_result[2]}')
    print(f'Number of Observations Used : {adfuller_result[3]}')
    if adfuller_result[1] < 0.05 :
        print('\nResult : The Time Series is Stationary.')
    else :
        print('\nResult : The Time Series is Non-Stationary.')
stationarity(dataset['AEP_MW'])

# AutoCorrelation Function ( ACF ) Analysis
print('\n======= AutoCorrelation Function ( ACF ) Chart =======')
def acf_plot (series) :
    plt.figure(figsize = (12,5))
    plot_acf(series,lags = 50)
    plt.title('ACF Chart')
    plt.show()
acf_plot(dataset['AEP_MW'])

# Partial AutoCorrelation Function ( PACF ) Analysis
print('\n======= Partial AutoCorrelation Function ( PACF ) Chart =======')
def pacf_plot (series) :
    plt.figure(figsize = (12,5))
    plot_pacf(series,lags = 50)
    plt.title('PACF Chart')
    plt.show()
pacf_plot(dataset['AEP_MW'])

# Seasonal Differencing
seasonal_diff = dataset['AEP_MW'].diff(24).dropna()
print('\n======= Seasonal AutoCorrelation Function ( ACF ) Chart =======')
acf_plot(seasonal_diff)
print('\n======= Seasonal Partial AutoCorrelation Function ( PACF ) Chart =======')
pacf_plot(seasonal_diff)

# Train/Test Split
print('\n======= Splitting Train/Test Datas =======')
train_size = int(len(dataset) * 0.8)
train_data = dataset.iloc[:train_size]
test_data = dataset.iloc[train_size:]
small_sample = train_data['AEP_MW'].iloc[-5000:]

# Comparing Models
print('\n======= Comparing Sample Models =======')
order_list = [(1,0,1),(2,0,1),(2,0,2),(3,0,2),(3,0,3),(4,0,3),(4,0,4)]
seasonal_order_list = [(1,1,1,24),(2,1,1,24),(2,1,2,24)]
count = 1
for seasonal_order in seasonal_order_list :
    for order in order_list :
        model_test = SARIMAX(small_sample,order = order,seasonal_order = seasonal_order,enforce_stationarity = False,enforce_invertibility = False)
        result_test = model_test.fit(maxiter = 50,disp = False)
        print(f'Model {count} AIC :',result_test.aic)
        print(f'Model {count} BIC :',result_test.bic)
        count += 1

# Best Model Selection
print('\n======= Selecting Best Model =======')
best_model = SARIMAX(small_sample,order = (2,0,1),seasonal_order = (2,1,1,24),enforce_stationarity = False,enforce_invertibility = False)
best_result = best_model.fit(maxiter = 200, disp = False)
print(best_result.summary())
print(best_result.mle_retvals)

# Training Best Model
print('\n======= Training Best Model =======')
best_model_training = SARIMAX(train_data["AEP_MW"],order = (2,0,1),seasonal_order = (2,1,1,24),enforce_stationarity = False,enforce_invertibility = False)
best_training_result = best_model_training.fit(maxiter = 300,low_memory = True,disp = False)

# Forecasting Best Model
print('\n======= Forecasting Best Model =======')
forecast = best_training_result.forecast(steps = len(test_data))

# Best Model Evaluation
print('\n======= Evaluating Best Model =======')
mae = mean_absolute_error(test_data['AEP_MW'],forecast)
rmse = np.sqrt(mean_squared_error(test_data['AEP_MW'],forecast))
mape = np.mean(np.abs((test_data['AEP_MW'] - forecast) / test_data['AEP_MW'])) * 100
print('MAE :',mae)
print('RMSE :',rmse)
print('MAPE :',mape)

# Forecast and Actual Comparing Chart
print('\n======= Forecast and Actual Comparing Chart =======')
hours = 24 * 7
def forecast_actual_plot () :
    plt.figure(figsize = (15,5))
    plt.plot(test_data.index[:hours],test_data['AEP_MW'][:hours],label = 'Actual')
    plt.plot(forecast.index[:hours],forecast[:hours],label = 'Forecast')
    plt.title("Actual vs Forecast")
    plt.xlabel("Datetime")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()
forecast_actual_plot()

# Building Final Model
print('\n======= Building Final Model =======')
final_model = SARIMAX(dataset['AEP_MW'],order = (2,0,1),seasonal_order = (2,1,1,24),enforce_stationarity = False,enforce_invertibility = False)
final_result = final_model.fit(maxiter = 300, low_memory = True, disp = False)
with open('AEP-SARIMA-Model.pkl', 'wb') as file:
    pickle.dump(final_result, file)
print('\nModel Saved Successfully !')
print(final_result.summary())

# Final Forecasting
print('\n======= Forecasting The Next Week =======')
forecast_steps = 24 * 7
future_forecast = final_result.forecast(steps = forecast_steps)
print(future_forecast)

# Final Forecasting DataFrame
print('\n======= Forecasting DataFrame =======')
future_dates = pd.date_range(start = dataset.index[-1] + pd.Timedelta(hours = 1),periods = forecast_steps,freq = 'h')
future_result = pd.DataFrame({'Datetime': future_dates,'Forecast_MW': future_forecast.values})
print(future_result)
future_result.to_csv('AEP-One-Week-Forecast.csv',index = False)
print('\nForecasting DataFrame Saved Successfully !')