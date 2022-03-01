import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load Data

company = 'AAPL'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data =  web.DataReader(company,'yahoo',start, end)
values = data['Close'].values.reshape(-1,1)
intraday_variation = values[1:-1]/values[0:-2]-1

print("ok")
#Prepare Data

scaler = MinMaxScaler(feature_range=(-1,1))

scaled_data = scaler.fit_transform(intraday_variation)

prediction_days = 60

x_train = []
y_train = []

for d in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[d-prediction_days:d,0])
    y_train.append(scaled_data[d,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the model

model = Sequential()

model.add( LSTM(units = 50 ,  return_sequences = True, input_shape=(x_train.shape[1],1))    )
model.add(Dropout(0.2))
model.add(LSTM(units = 50 ,  return_sequences = True ))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1)) #prediction of day d+1

model.compile(optimizer = 'adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs = 25, batch_size = 32)

'''Test the model accuracy on existing data'''
#Load test data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company,'yahoo',test_start, test_end)
actual_prices = test_data['Close'].values
actual_prices = actual_prices.reshape(-1,1)
actual_variations = actual_prices[1:-1]/ actual_prices[0:-2]-1

total_dataset = pd.concat((data['Close'], test_data['Close']))

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = model_inputs[1:-1]/model_inputs[0:-2]-1
model_inputs = scaler.transform(model_inputs)

#Make prediction on test Data
x_test = []
for d in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[d-prediction_days:d,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_variations = model.predict(x_test)
predicted_variations = scaler.inverse_transform(predicted_variations)

#Plot the test prediction
plt.plot(actual_variations, color="black", label = f"Actual {company} price")
plt.plot(predicted_variations, color = "green", label = f"Predicted {company} price")

plt.title(f"{company} price")
plt.xlabel("time")
plt.ylabel(f"{company} share price")
plt.legend()

plt.show()