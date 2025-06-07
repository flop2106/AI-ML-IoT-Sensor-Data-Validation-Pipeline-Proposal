
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


def generate_historical_old_sensor_data():
    sensor_ids = ['old_sensor_1', 'old_sensor_2']
    data = []
    start_time = datetime.now() - timedelta(seconds=1000*5)
    for sensor_id in sensor_ids:
        for i in range(500):
            row = {
                'sensor_id': sensor_id,
                'temperature': np.random.normal(75, 5),
                'vibration': np.random.normal(1.5, 0.2),
                'wind_speed': np.random.normal(12, 3),
                'humidity': np.random.normal(60, 10),
                'timestamp': start_time + timedelta(seconds=i*5)
            }
            data.append(row)
    df = pd.DataFrame(data)
    df.to_csv('old_sensor_data.csv', index=False)

def generate_initial_new_sensor_data():
    sensor_ids = ['new_sensor_1', 'new_sensor_2', 'new_sensor_3']
    data = []
    start_time = datetime.now() - timedelta(seconds=100*5)
    for sensor_id in sensor_ids:
        for i in range(100):
            vibration = np.random.normal(1.5, 0.2)
            temperature = np.random.normal(75, 5)
            wind_speed = np.random.normal(12, 3)
            humidity = np.random.normal(60, 10)

            if sensor_id == 'new_sensor_2':
                vibration += i * 0.01
                temperature += i * 0.02

            if sensor_id == 'new_sensor_3' and i > 60:
                vibration += 1.5
                temperature += 10
                wind_speed += 5
                humidity += 15

            row = {
                'sensor_id': sensor_id,
                'temperature': temperature,
                'vibration': vibration,
                'wind_speed': wind_speed,
                'humidity': humidity,
                'timestamp': start_time #+ timedelta(seconds=i*6)
            }
            data.append(row)
    df = pd.DataFrame(data)
    df.to_csv('new_sensor_data.csv', index=False)

def append_new_row():
    sensor_ids = ['new_sensor_1', 'new_sensor_2', 'new_sensor_3']
    for i in range(6):
        time.sleep(5)
        new_rows = []
        timestamp = datetime.now()
        for sensor_id in sensor_ids:
            vibration = np.random.normal(1.5, 0.2)
            temperature = np.random.normal(75, 5)
            wind_speed = np.random.normal(12, 3)
            humidity = np.random.normal(60, 10)

            if sensor_id == 'new_sensor_2':
                vibration += i * 0.02
                temperature += i * 0.03
            if sensor_id == 'new_sensor_3':
                vibration += 5.0
                temperature += 20
                wind_speed += 10
                humidity += 30

            new_rows.append({
                'sensor_id': sensor_id,
                'temperature': temperature,
                'vibration': vibration,
                'wind_speed': wind_speed,
                'humidity': humidity,
                'timestamp': timestamp
            })

        df_new = pd.DataFrame(new_rows)
        df_new.to_csv('new_sensor_data.csv', mode='a', header=False, index=False)
        print(f"{timestamp}: New Data Created! ")
        

if __name__=="__main__":
    pass

