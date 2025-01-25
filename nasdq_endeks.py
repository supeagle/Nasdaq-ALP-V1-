# -*- coding: utf-8 -*-

pip install prophet

!pip install -q kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install keras-tuner

!pip install pmdarima

import kagglehub
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras_tuner import RandomSearch
from prophet.make_holidays import make_holidays_df
from pmdarima import auto_arima

path = kagglehub.dataset_download("sai14karthik/nasdq-dataset")

print("Path to dataset files:", path)

df = pd.read_csv("/root/.cache/kagglehub/datasets/sai14karthik/nasdq-dataset/versions/1/nasdq.csv")
df.head()

df.info()

df.describe()

df['Date'] = pd.to_datetime(df['Date'])

prophet_data = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

# Train the Prophet model with improved parameters
prophet_model = Prophet(
    changepoint_prior_scale=0.05,  # More sensitive to trend changes
    seasonality_prior_scale=10.0,  # Adjust seasonality strength
    yearly_seasonality=True,       # Enable yearly seasonality
    weekly_seasonality=True        # Enable weekly seasonality
)
prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Add monthly seasonality
prophet_model.fit(prophet_data)

# Make future predictions
future_dates = prophet_model.make_future_dataframe(periods=365)
forecast = prophet_model.predict(future_dates)
forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast = forecast.rename(columns={'ds': 'Date'})

# Merge actual and forecast data
result = pd.merge(df[['Date', 'Close']], forecast, on='Date', how='outer')
print(result.head())

# Evaluate the model
actual = prophet_data['y'].values
predicted = forecast['yhat'][:len(actual)].values
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(18, 6))
sns.lineplot(x="Date", y="Close", data=result, label="Gerçek Değerler", linestyle="--", color="green")
sns.lineplot(x="Date", y="yhat", data=result, label="Tahmin Değerler", color="blue")
plt.fill_between(result['Date'], result['yhat_lower'], result['yhat_upper'], color="blue", alpha=0.2, label="Tahmin Aralığı")
plt.title("Prophet ile Endeks Fiyat Tahmini", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Fiyat", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

df.set_index('Date', inplace=True)

close_prices = df['Close']

train_size = int(len(close_prices) * 0.8)
train_data = close_prices[:train_size]
test_data = close_prices[train_size:]

p, d, q = 5, 1,3
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test_data))
forecast_index = test_data.index

mae = mean_absolute_error(test_data, forecast)
rmse = np.sqrt(mean_squared_error(test_data, forecast))
r2 = r2_score(test_data, forecast)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

plt.figure(figsize=(14, 7))
sns.lineplot(x=train_data.index, y=train_data.values, label="Eğitim Verisi", color="blue")
sns.lineplot(x=results["Date"], y=results["Actual"], label="Gerçek Değerler", color="green", linestyle="--")
sns.lineplot(x=results["Date"], y=results["Forecast"], label="Tahmin Değerler", color="red")
plt.title("ARIMA ile Endeks Fiyat Tahmini ", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Fiyat", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60  # Son 60 günün verisi ile tahmin yap
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=150, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)
optimizer = Adam(learning_rate=0.001)

# Tahmin
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_actual, predicted)
rmse = np.sqrt(mean_squared_error(y_test_actual, predicted))
print(f"MAE: {mae}, RMSE: {rmse}")

r2 = r2_score(y_test_actual, predicted)
print(f"Test R-squared: {r2 * 100:.2f}%")

plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test_actual):], y_test_actual, label="Gerçek Değerler", linestyle="--")
plt.plot(df.index[-len(predicted):], predicted, label="Tahmin Değerler")
plt.title("LSTM ile Endeks Fiyat Tahmini")
plt.xlabel("Tarih")
plt.ylabel("Fiyat")
plt.legend()
plt.show()

# Volatilite Tahminleri
df['Volatility'] = df['Close'].pct_change().rolling(window=5).std()
df = df.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Volatility'].values.reshape(-1, 1))

seq_length = 60  # Son 60 gün verisi
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# LSTM giriş boyutunu ayarlama
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=150, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=200, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=12, batch_size=32)
optimizer = Adam(learning_rate=0.0012)
# Tahmin
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_actual, predicted)
rmse = np.sqrt(mean_squared_error(y_test_actual, predicted))
print(f"MAE: {mae}, RMSE: {rmse}")

r2 = r2_score(y_test_actual, predicted)
print(f"Test R-squared: {r2 * 100:.2f}%")

results = pd.DataFrame({
    "Date": df.index[-len(y_test_actual):],
    "Actual Volatility": y_test_actual.flatten(),
    "Predicted Volatility": predicted.flatten()
})

plt.figure(figsize=(14, 7))
sns.lineplot(x="Date", y="Actual Volatility", data=results, label="Gerçek Volatilite", linestyle="--", color="blue")
sns.lineplot(x="Date", y="Predicted Volatility", data=results, label="Tahmin Edilen Volatilite", color="red")
plt.title("LSTM ile Volatilite Tahmini", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Volatilite", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

df["Volatility"] = df["Gold"].pct_change() * 100
volatility_data = df.dropna(subset=["Volatility"])[["Date", "Volatility"]]
volatility_data = volatility_data.rename(columns={"Date": "ds", "Volatility": "y"})

df["Gold_Return"] = df["Gold"].pct_change()
df["Gold_Volatility"] = df["Gold_Return"].rolling(window=20).std()
df = df.dropna()

# Prepare data for Prophet model
volatility_data = df[["Date", "Gold_Volatility"]].rename(columns={"Date": "ds", "Gold_Volatility": "y"})

# Train a simple Prophet model
volatility_model = Prophet()
volatility_model.fit(volatility_data)

# Make future predictions
future = volatility_model.make_future_dataframe(periods=30)
forecast = volatility_model.predict(future)

# Evaluate the model
actual = volatility_data['y'].values
predicted = forecast['yhat'][:len(actual)].values
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

forecast_volatility = forecast.merge(volatility_data, on="ds", how="left", suffixes=("", "_actual"))

# Plot actual vs predicted values with uncertainty intervals
plt.figure(figsize=(14, 7))
sns.lineplot(x="ds", y="yhat", data=forecast_volatility, label="Öngörülen Volatilite", linewidth=2)
sns.lineplot(x="ds", y="y", data=forecast_volatility, label="Gerçek Volatilite", linewidth=2, alpha=0.7)
plt.fill_between(
    forecast_volatility["ds"],
    forecast_volatility["yhat_lower"],
    forecast_volatility["yhat_upper"],
    color="blue",
    alpha=0.2,
    label="Belirsizlik Aralığı"
)
plt.title("Altın İçin Öngörülen ve Gerçek Volatilite (Prophet Modeli)", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Volatilite (%)", fontsize=12)
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x="ds", y="yhat", data=forecast_volatility, label="Öngörülen Volatility", linewidth=2)
sns.lineplot(x="ds", y="actual", data=forecast_volatility, label="Güncel Volatility", linewidth=2, alpha=0.7)
plt.fill_between(
    forecast_volatility["ds"],
    forecast_volatility["yhat_lower"],
    forecast_volatility["yhat_upper"],
    color="blue",
    alpha=0.2,
    label="Uncertainty Interval"
)
plt.title("Altın İçin Öngörülen Volatilite (Prophet)", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Volatilite (%)", fontsize=12)
plt.legend()
plt.show()

X = df[["InterestRate", "ExchangeRate", "Oil", "VIX", "TEDSpread", "EFFR"]]  # Bağımsız değişkenler
y = df["Gold"]  # Bağımlı değişken

# Eksik verileri kontrol et
if X.isnull().sum().any() or y.isnull().sum().any():
    print("Eksik veriler temizleniyor...")
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train R2: {train_r2}, Test R2: {test_r2}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")

# Sonuçları görselleştirme
plt.figure(figsize=(14, 7))
sns.scatterplot(x=y_test, y=y_test_pred, label="Predicted vs Actual", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', label="Ideal Fit")
plt.title("Actual vs Predicted Gold Prices", fontsize=16)
plt.xlabel("Actual Gold Prices", fontsize=12)
plt.ylabel("Predicted Gold Prices", fontsize=12)
plt.legend()
plt.show()

x = df[["InterestRate", "ExchangeRate", "Oil", "VIX", "TEDSpread", "EFFR"]]
y = df["Gold"]

# Eksik verileri temizle
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge modeli ve parametre aralığı
ridge = Ridge()
param_grid = {
    "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularization güçleri
}

# Grid Search CV
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring="r2", cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# En iyi model ve parametreler
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Performans değerlendirmesi
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train R2: {train_r2}, Test R2: {test_r2}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")

variables = ["Gold", "InterestRate", "ExchangeRate", "Oil", "VIX", "TEDSpread", "EFFR"]

# Korelasyon matrisi
correlation_matrix = df[variables].corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Altın için Korelasyon Haritası ", fontsize=16)
plt.show()

df["Gold_Return"] = df["Gold"].pct_change()
df["Gold_Volatility"] = df["Gold_Return"].rolling(window=20).std()
df = df.dropna()

# Select features and target
features = df[["Gold", "Gold_Return", "InterestRate", "ExchangeRate", "VIX", "TEDSpread", "EFFR"]]
target = df["Gold_Volatility"]

# Scale features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare time series data
time_steps = 10
X, y = [], []
for i in range(time_steps, len(features_scaled)):
    X.append(features_scaled[i - time_steps:i])
    y.append(target.iloc[i])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(LSTM(
        units=hp.Int('units2', min_value=32, max_value=128, step=32),
        return_sequences=True
    ))
    model.add(Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(LSTM(
        units=hp.Int('units3', min_value=16, max_value=64, step=16)
    ))
    model.add(Dropout(hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error'
    )
    return model

# Initialize Keras Tuner
import keras_tuner as kt

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='gold_volatility_lstm'
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# Build and train the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the best model
best_model.save("gold_volatility_lstm_best_model.h5")

# Evaluate the best model
evaluation = best_model.evaluate(X_test, y_test, verbose=1)
print("Test Loss:", evaluation)

# Predict and compare predictions
y_pred = best_model.predict(X_test)
comparison = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
print(comparison.head())

# Calculate RMSE, MAE, and R2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(12, 6))
plt.plot(comparison['Actual'], label='Güncel Volatilite', alpha=0.7)
plt.plot(comparison['Predicted'], label='Öngörülen Volatility', alpha=0.7)
plt.title('Güncel ve Öngörülen Volatility ')
plt.xlabel('Gün')
plt.ylabel('Volatilite')
plt.legend()
plt.show()
