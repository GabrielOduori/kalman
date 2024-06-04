import numpy as np
import unittest


# intial_v : location
# intial_x : speed estimate

NUMVARS = 2
class KF:
    def __init__(self, initial_v: float, 
                 initial_x: float,
                 acc_variance: float): # Add type annotations to make clear what is expected
        #Mean of the state GRV
        self._x = np.array([initial_x, initial_v])
        self.acc_variance = acc_variance

        #Covariance of the state GRV
        self._P = np.eye(2)
        #Mean of the state GRV

        self.x = np.array([initial_x, initial_v])

        #Covariance of the state GRV

        self.P = np.eye(2)
    def predict(self, dt: float) -> None:


        # x = F x

        # P = F P F^T + Q Qt a 
                
        #State transition matrix
        F = np.array([[1, dt],
                      [0, 1]])
        new_x = F.dot(self._x)
        
        #Process noise covariance
         
        Q = np.array([0.5 * dt**2, dt]).reshape((2, 1))
        #Predict the state

        #Predict the state covariance

        new_P = F.dot(self._P).dot(F.T) + Q.dot(Q.T) * self.acc_variance

        self._x = new_x
        self._P = new_P
        
    def update(self,maes_value:float, meas_variance:float):
        # y = z - H x
        # S = H P H^T + R
        # K = P H^T S^-1
        # x = x + K y
        # P = (I - K H) P

        #Measurement matrix

        H = np.zeros((1, 2))

        z = np.array([maes_value])
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
        return self._x[0]
    
    @property
    def vel(self) -> float:
        return self._x[1]
    


