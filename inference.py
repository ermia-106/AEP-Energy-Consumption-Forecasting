import sys
import pickle

sys.stdout.write('Hi, I\'m ermia-106 .\nI\'m Here to Forecast Energy Consumption in America,\n')
user_hours = int(input('So Enter The Number of Hours for Forecasting from 2018-08-03 00:00:00 Onwards : '))

if user_hours == 0 :
    print('0 isn\'t a Valid Number for Hour.')
elif user_hours == 1 :
    print('======= Forecasting The Next 1 Hour =======')
else :
    print(f'======= Forecasting The Next {user_hours} Hours =======')

with open(r'E:\Programming\GitHub Repository\Python\AEP-Energy-Consumption-Forecasting\AEP-SARIMA-Model.pkl', 'rb') as file:
    model = pickle.load(file)
forecast = model.forecast(steps = user_hours)
print(forecast)