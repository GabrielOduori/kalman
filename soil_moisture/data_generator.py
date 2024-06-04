import numpy as np
import pandas as pd


def generate_sythetic_data():
    np.random.seed(0)
    n_days = 180 # 6 months
    

    true_values = np.linspace(0, 10, n_days)
    sensor_noise = np.random.normal(0, 1, n_days)
    satellite_noise = np.random.normal(0, 0.5, n_days//30)

    sensor_measurements = true_values + sensor_noise

    satellite_measurements = np.full(n_days, np.nan)


    for i in range(0, n_days, 30):
        satellite_measurements[i] = true_values[i] + satellite_noise[i//30]


    sensor_data = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2023', periods=n_days, freq='D'),
        'Sensor Measurements': sensor_measurements,
    })


    satellite_data = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2023', periods=n_days, freq='D'),
        'Satellite Measurements': satellite_measurements,
    })


    return sensor_data, satellite_data, true_values
  