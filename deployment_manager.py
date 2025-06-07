
from sensor_data_generator import generate_historical_old_sensor_data, generate_initial_new_sensor_data, append_new_row
from eda_cleaning import perform_eda_and_cleaning
from model_handler import train_model, predict, train_and_evaluate_model
import pandas as pd
import time

def initial_deployment():
    generate_historical_old_sensor_data()
    generate_initial_new_sensor_data()
    perform_eda_and_cleaning("old_sensor_data.csv")
    train_model()
    perform_eda_and_cleaning("new_sensor_data.csv")
    predict()
    

def run_live_prediction_and_retrain():
    #generate_initial_new_sensor_data()
    for i in range(360):  # Simulate 360 rounds (30 minutes)
        append_new_row()
        perform_eda_and_cleaning(input_csv='new_sensor_data.csv')
        predict()
        print("Complete Prediction")
        df = pd.read_csv('new_sensor_data.csv')
        if i % 10 == 0:
            print("Re-Train Model")
            #df_old = pd.read_csv("old_sensor_data.csv")
            #df_old.to_csv("train_data.csv", index = False)
            #df.to_csv("train_data.csv", mode = 'a', header=False, index= False)
            perform_eda_and_cleaning()
            train_model()

if __name__ == "__main__":
    #initial_deployment()
    run_live_prediction_and_retrain()