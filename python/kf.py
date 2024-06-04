import numpy as np
import unittest


# intial_v : location
# intial_x : speed estimate
# acc_variance : variance of the acceleration

ix = 0
iv = 1
NUMVARS = ix + 1


NUMVARS = 2
class KF:
    def __init__(self, initial_x: float, 
                 initial_v: float,
                 acc_variance: float): # Add type annotations to make clear what is expected
        
        #Mean of the state GRV

        self._x = np.zeros(NUMVARS)

        self._x[ix] = initial_x
        self._x[iv] = initial_v
        
        self._acc_variance = acc_variance

        #Covariance of the state GRV
        #Mean of the state GRV

        #Covariance of the state GRV
        self._P = np.eye(NUMVARS)

    def predict(self, dt: float) -> None:


        # x = F x

        # P = F P F^T + Q Qt a 
                
        #State transition matrix

        F = np.eye(NUMVARS)
        F[ix, iv] = dt

        # F = np.array([[1, dt],
        #               [0, 1]])
        new_x = F.dot(self._x)
        
        #Process noise covariance
        Q = np.zeros((2,1))
        Q[ix] = 0.5 * dt**2
        Q[iv] = dt
        # Q = np.array([0.5 * dt**2, dt]).reshape((2, 1))
        #Predict the state

        #Predict the state covariance

        new_P = F.dot(self._P).dot(F.T) + Q.dot(Q.T) * self._acc_variance

        self._P = new_P
        self._x = new_x
        
    def update(self, meas_value:float, meas_variance:float):
        # y = z - H x y is the innovation
        # S = H P H^T + R S is the innovation covariance
        # K = P H^T S^-1 K is the Kalman gain
        # x = x + K y x is the new state estimate
        # P = (I - K H) P P is the new state covariance

        #Measurement matrix

        H = np.zeros((1, 2))

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P


    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x
     
    @property
    def pos(self) -> float:
        return self._x[ix]
    
    @property
    def vel(self) -> float:
        return self._x[iv]
    


