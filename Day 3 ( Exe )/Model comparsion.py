import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

solar_data_path = 'D:/csv/solar data edited.csv'
solar_data = pd.read_csv(solar_data_path)

print(solar_data.columns)

solar_data['Datetime'] = pd.to_datetime(solar_data['DATE &TIME'], dayfirst=True)
solar_data.drop(columns=['DATE &TIME'], inplace=True)
solar_data.ffill(inplace=True)

solar_features = solar_data.drop(columns=['Datetime'])
solar_scaler = StandardScaler()
solar_features_scaled = solar_scaler.fit_transform(solar_features)


solar_data_scaled = pd.DataFrame(solar_features_scaled, columns=solar_features.columns)
solar_data_scaled['Datetime'] = solar_data['Datetime']


def extract_datetime_features(df, datetime_col):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['hour'] = df[datetime_col].dt.hour
    df['day'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    df['year'] = df[datetime_col].dt.year
    df['dayofweek'] = df[datetime_col].dt.dayofweek
    return df

solar_data_scaled = extract_datetime_features(solar_data_scaled, 'Datetime')
solar_data_scaled.drop(columns=['Datetime'], inplace=True)


solar_data_scaled['source'] = 'solar'
solar_data_scaled.columns = [f'solar_{col}' if col != 'source' else col for col in solar_data_scaled.columns]


combined_data = solar_data_scaled
combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
combined_data_filled = combined_data.fillna(0)


solar_target = 'solar_ENERGY GENERATION (KWH)'


x_solar = combined_data_filled.drop(columns=[solar_target, 'source'])
y_solar = combined_data_filled[solar_target]
x_train_solar, x_test_solar, y_train_solar, y_test_solar = train_test_split(x_solar, y_solar, test_size=0.2, random_state=42)


def reshape_for_lstm(x):
    return np.reshape(x, (x.shape[0], x.shape[1], 1))

x_train_solar_lstm = reshape_for_lstm(x_train_solar)
x_test_solar_lstm = reshape_for_lstm(x_test_solar)


def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train_solar_lstm.shape[1], 1)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_nn_model():
    model = Sequential()
    model.add(Dense(64, input_dim=x_train_solar.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


class KerasRegressorWrapper:
    def __init__(self, build_fn):
        self.build_fn = build_fn
        self.model = None
        
    def fit(self, x, y, epochs=20, batch_size=16, verbose=0):
        self.model = self.build_fn()
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
    def predict(self, x):
        return self.model.predict(x)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'Neural Network': KerasRegressorWrapper(build_fn=create_nn_model),
    'LSTM': KerasRegressorWrapper(build_fn=create_lstm_model),
    'Bagging Regressor': BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=42),
}


def evaluate_models(x_train, x_test, y_train, y_test):
    results = []
    for name, model in models.items():
        if name == 'LSTM':
            model.fit(reshape_for_lstm(x_train), y_train, epochs=20, batch_size=16, verbose=0)
            y_pred = model.predict(reshape_for_lstm(x_test))
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))
        print(f"{name}: MSE={mse:.2f}, R2={r2:.2f}")
    return results

print("\nSolar Data Model Evaluation")
evaluate_models(x_train_solar, x_test_solar, y_train_solar, y_test_solar)
