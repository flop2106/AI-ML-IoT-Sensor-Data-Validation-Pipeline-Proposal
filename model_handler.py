
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

def train_model():
    df = pd.read_csv('cleaned_data.csv')
    df = df[(df["sensor_id"]!= "new_sensor_2") & (df["sensor_id"]!= "new_sensor_3")]
    X = df.drop(columns=['sensor_id', 'timestamp'])

    # Train Isolation Forest
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(X)
    joblib.dump(iso_model, 'model_isolation.pkl')

    # Train XGBoost models (one per feature)
    features = ['temperature', 'vibration', 'wind_speed', 'humidity']
    xgb_models = {}
    for feature in features:
        X_train = X.drop(columns=[feature])
        y_train = X[feature]
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        xgb_models[feature] = model
    joblib.dump(xgb_models, 'model_xgb.pkl')

    # Train Autoencoder
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    ae_model = MLPRegressor(hidden_layer_sizes=(12, 6, 12), max_iter=500, random_state=42)
    ae_model.fit(X_scaled, X_scaled)
    joblib.dump(ae_model, 'model_autoencoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

def predict():
    df = pd.read_csv('cleaned_data.csv')
    X = df.drop(columns=['sensor_id', 'timestamp'])

    # Load models
    iso_model = joblib.load('model_isolation.pkl')
    xgb_models = joblib.load('model_xgb.pkl')
    ae_model = joblib.load('model_autoencoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Isolation Forest prediction
    df['iso_flag'] = iso_model.predict(X)

    # XGBoost residuals
    xgb_residuals = []
    features = ['temperature', 'vibration', 'wind_speed', 'humidity']
    for feature in features:
        X_test = X.drop(columns=[feature])
        y_true = X[feature]
        y_pred = xgb_models[feature].predict(X_test)
        residual = np.abs(y_true - y_pred)
        xgb_residuals.append(residual)
    df['xgb_residual'] = np.mean(xgb_residuals, axis=0)
    xgb_thresh = df['xgb_residual'].mean() + 2 * df['xgb_residual'].std()
    df['xgb_flag'] = df['xgb_residual'] > xgb_thresh

    # Autoencoder prediction
    X_scaled = scaler.transform(X)
    ae_pred = ae_model.predict(X_scaled)
    ae_error = np.mean(np.square(X_scaled - ae_pred), axis=1)
    ae_thresh = np.mean(ae_error) + 2 * np.std(ae_error)
    df['ae_error'] = ae_error
    df['ae_flag'] = ae_error > ae_thresh

    # Combine flags into final status
    df['status'] = 'Normal'    
    df.loc[(df['xgb_flag'] | df['ae_flag']) & (df['iso_flag'] != -1), 'status'] = 'Warning'
    df.loc[(df['xgb_flag'] & df['ae_flag']), 'status'] = 'Faulty'
    df.loc[df['iso_flag'] == -1, 'status'] = 'Faulty'

    df.to_csv('predictions.csv', index=False)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_model():
    df = pd.read_csv('cleaned_data.csv')
    X = df.drop(columns=['sensor_id', 'timestamp'])

    # Simulate label generation: Normal = 1, Faulty = -1
    # For demo, label 5% as anomalies randomly
    np.random.seed(42)
    y = np.ones(len(X))
    y[np.random.choice(len(X), size=int(0.05*len(X)), replace=False)] = -1

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    preds = model.predict(X_test)

    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, pos_label=-1))
    print("Recall:", recall_score(y_test, preds, pos_label=-1))
    print("F1 Score:", f1_score(y_test, preds, pos_label=-1))