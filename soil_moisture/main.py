import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from kalman_filter import KalmanFilter
from data_generator import generate_sythetic_data


plt.ion()
plt.figure()

sensor_data, satellite_data, true_values = generate_sythetic_data()
initial_x = float(sensor_data['Sensor Measurements'][0])
accel_covariance = 0.1
# initial_x = sensor_data['Sensor Measurement'][0]


kf = KalmanFilter(initial_x, accel_covariance)

# Merge the sensor and satellite data

data = sensor_data.merge(satellite_data, on='Date', how='left')

sensor_variance = 1.0


# Apply the Kalman filter to the data

for index, row in data.iterrows():
    if not np.isnan(row['Satellite Measurements']):
        kf.update(row['Satellite Measurements'], sensor_variance)
    else:
        kf.update(row['Sensor Measurements'], sensor_variance) # Update the sensor measurements if there is no satellite measurement
    data.at[index, 'Satellite Measurements'] = kf.mean



    # Evaluate results
    estimated_values = data['Satellite Measurement'].values
    mae = mean_absolute_error(true_values, estimated_values)
    rmse = np.sqrt(mean_squared_error(true_values, estimated_values))
    r2 = r2_score(true_values, estimated_values)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")




# Plot the results

    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], true_values, label='True Values', linewidth=2)
    plt.plot(data['Date'], data['Sensor Measurements'], label='Sensor Measurements', alpha=0.6)
    plt.plot(data['Date'], data['Satellite Measurements'], label='Estimated Satellite Measurements', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Kalman Filter: Sensor vs Satellite Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.ginput(1)
    plt.ginput(1, timeout=0)


