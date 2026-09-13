import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter_Pair():
    def __init__(self, asset1, asset2, process_noise_wt=1e-5, measurment_noise_vt=1e-3):
        self.asset1 = asset1
        self.asset2 = asset2
        self.process_noice_wt = process_noise_wt
        self.measurment_noise_vt = measurment_noise_vt
        # Kalman Variables
        self.beta = 0.0                      # Current hedge ratio (state)
        self.P = 1.0                         # Error covariance
        self.Q = self.process_noice_wt       # Process noise
        self.R = self.measurment_noise_vt    # Measurement noise
        # Historical data storage
        self.prices1 = []
        self.prices2 = []
        self.betas = []
        self.spreads = []
    
    def __repr__(self):
        return str({'class': "KalmanFilter_Pair", 'asset_1': self.asset1, 'asset_2': self.asset2, 'process_noise': self.process_noice_wt, 'measurment_noise': self.measurment_noise_vt})
    
    def correction(self, price1, price2):     # price1 is dependent, meanwhile price2 is independent
        # Prediction of beta and error cov.
        beta_pred = self.beta     # best prediction state prediction is to use previous right prediction as current new prediction
        P_pred = self.P + self.Q  # predicted uncertainty
        kalman_gain_Gt = (P_pred * price2) / (((price2**2) * P_pred) + self.R)
        # correction step
        self.beta = beta_pred + (kalman_gain_Gt * (price1 - (beta_pred * price2)))
        self.P = P_pred * (1 - (kalman_gain_Gt * price2))
        # spread & signal
        spread = price1 - (beta_pred * price2)
        self.spreads.append(spread)
        # Forecast Error variance
        F = (price2**2) * P_pred + self.R
        signal = self.get_signal(spread, F)
        return spread, signal
    
    def get_signal(self, spread, var):
        threshold = 2 * np.sqrt(var)
        if spread > threshold:
            return -1  # Short spread (sell Y, buy X)
        elif spread < -threshold:
            return 1   # Long spread (buy Y, sell X)
        else:
            return 0   # No position
        
class KalmanFilter_multivariate():
    def __init__(self, n_factors, process_noise=1e-5, measurment_noise=1e-3):
        self.n = n_factors
        self.process_noise = process_noise
        self.measurment_noise = measurment_noise
        self.beta = np.zeros(n_factors)       # creating matrix of beta with dimensions of n_factors
        self.P = np.eye(n_factors) * 1000     # initial uncertainty
        self.Q = np.eye(n_factors) * process_noise
        self.R = measurment_noise
    
    def __repr__(self):
        return str({'class': 'KalmanFilter_multivariate', 'n_factor': self.n, 'process noise wt': self.process_noise, 'measurment noise vt': self.measurment_noise})
    
    def correction(self, y, x):       # y: dependent variable (scalar); x: Independent vector (n_factors,)
        x = np.asarray(x).flatten()
        beta_pred = self.beta
        P_pred = self.P + self.Q
        F = float(x.T @ P_pred @ x + self.R)
        Kalman_gain_Gt = (P_pred @ x) / F
        self.beta = beta_pred + Kalman_gain_Gt * (y - float(beta_pred.T @ x))
        self.P = (np.eye(self.n) - np.outer(Kalman_gain_Gt, x)) @ P_pred
        spread = y - float(self.beta.T @ x)
        return spread, self.beta
