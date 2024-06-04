import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KalmanFilter:

    

    def __init__(self, intial_x:float, accel_covariance:float) -> None:
        self._x = intial_x
        self._accel_covariance = accel_covariance
        self._P = 1.0 # initial covariance


    def predict() -> None:
        # state prediction

        # No need for this in our case
        """
        self._x = self._x
        self._P = self._P + self._accel_covariance
        """
        pass

        # state update

    def update(self, sensor_meas: float, sensor_covariance: float) -> None:
        # measurement update step

        # Kalman gain
        K = self._P / (self._P + sensor_covariance)

        # state update
        self._x = self._x + K * (sensor_meas - self._x)

        # covariance update
        self._P = (1 - K) * self._P

    @property
    def mean(self) -> float:
            return self._x
    
    # @property
    # def covariance(self) -> float:
    #     return self._P
        


