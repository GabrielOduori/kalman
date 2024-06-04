import unittest
import numpy as np
import pandas as pd

# Assuming the Kalman Filter implementation is in a module named kalman_filter
from kalman_filter import KalmanFilter

class TestKalmanFilter(unittest.TestCase):

    def setUp(self):
        # Create synthetic data for testing
        self.data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'Sensor Measurement': np.random.normal(loc=0.0, scale=1.0, size=30),
            'Satellite Measurement': [np.nan if i % 5 != 0 else np.random.normal(loc=0.0, scale=1.0) for i in range(30)]
        })

        # True values for testing (synthetic example)
        self.true_values = [self.data['Sensor Measurement'][i] for i in range(30) if i % 5 == 0]

        # Kalman Filter initial parameters
        self.A = np.eye(1)
        self.B = np.zeros((1, 1))
        self.H = np.eye(1)
        self.Q = np.eye(1) * 0.1
        self.R = np.eye(1) * 1.0

        # Initialize state estimate and covariance
        self.x_est = np.zeros((1, 1))
        self.P = np.eye(1)

    def test_kalman_filter(self):
        for index, row in self.data.iterrows():
            if not np.isnan(row['Satellite Measurement']):
                z = np.array([[row['Satellite Measurement']]])
                self.x_est, self.P = KalmanFilter(z, self.x_est, self.P, self.A, self.B, self.H, self.Q, self.R)
            else:
                z = np.array([[row['Sensor Measurement']]])
                self.x_est, self.P = KalmanFilter(z, self.x_est, self.P, self.A, self.B, self.H, self.Q, self.R)
                self.data.at[index, 'Satellite Measurement'] = self.x_est[0, 0]

        estimated_values = self.data['Satellite Measurement'][::5].values
        np.testing.assert_almost_equal(estimated_values, self.true_values, decimal=1)

if __name__ == '__main__':
    unittest.main()
