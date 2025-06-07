
import pandas as pd

def perform_eda_and_cleaning(input_csv='new_sensor_data.csv'):
    df = pd.read_csv(input_csv)
    df.dropna(inplace=True)
    for col in ['temperature', 'vibration', 'wind_speed', 'humidity']:
        df[f'{col}_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_std_5'] = df[col].rolling(window=5, min_periods=1).std()
        df[f'{col}_diff'] = df[col].diff().fillna(0)
    df.dropna(inplace=True)
    df.to_csv('cleaned_data.csv', index=False)
