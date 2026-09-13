import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from typing import Dict,Optional
 
class StationarityTest:
    def __init__(self,time_series:pd.DataFrame|pd.Series,lag_max:Optional[int]=None,regression_type:str="c",required_lag:str="AIC",regression_result:bool=False):
        """Required I/P to test stationarity
        Args:
            time_series:pd.DataFrame: Time series or data series which is to analyze
            lag_max:int: Number of max. lags can used, if None then default value is 12*(n_obs/100)^0.25
            regression_type:str:{"c","ct","ctt","n"} Constant and trend order to include in regression.
                "c":constant only (default).
                "ct":constant and trend.
                "ctt":constant, and linear and quadratic trend.
                "n":no constant, no trend. 
            required_lag:str:{"AIC", "BIC", "t-stat", None}
                Method to use when automatically determining the lag length among the values 0, 1, …, maxlag.
                ~ If “AIC” (default) or “BIC”, then the number of lags is chosen to minimize the corresponding information criterion.
                ~ "t-stat" based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag length is significant using a 5%-sized test.
                ~ If None, then the number of included lags is set to maxlag.
            regression_result:bool=False: If True, the full regression results are returned. Default is False
        O/P:
            {adf_statistics,p-value,used_lags,n_observation,{cv_1%,cv_5%,cv_10%},ic_best(if autolag is not None),res_store}"""
        
        self.TS=pd.DataFrame(time_series)
        self.maxlag=lag_max
        self.regression_type=regression_type
        self.required_lag=required_lag
        self.regression_result=regression_result
        self._validate_inputs()

    def __repr__(self):
        representation=str(f"class to test stationarity of Time series\nIP_TS:{self.TS.describe()}\nmaxlags:{self.maxlag},regression_type={self.regression_type}\nrequired_lag={self.required_lag}\nOP:")
        return representation
    
    def _validate_inputs(self):     # Check if any given input is wrong or not?
        valid_lag=["AIC","BIC","t-stat",None]
        valid_reg=["c","ct","ctt","n"]
        if self.required_lag not in valid_lag:
          raise ValueError(f"criteria must be one of {valid_lag}, got {self.required_lag}")
        elif self.TS.empty:
            raise ValueError("Data cannot be empty")
        elif self.TS.isnull().any().any() or np.isinf(self.TS.values).any():
            print("Warning: Data contain Nan or Inf values. Dropping Nan rows")
            self.TS=self.TS.dropna()
        elif len(self.TS)<20:
            raise ValueError('Length of data is to short, try to give more observations')
        elif self.regression_type not in valid_reg:
            raise ValueError (f"Regression type must be one of {valid_reg}, got {self.regression_type}")
        else:
            print('All input data are verified and are True for execution')

    def difference(self,order:int=1):
        """ ΔX_t = X_t - X_(t-1)
        Args:
            ~ order:int differencing order
        O/P: Differenced series """
        return self.TS.diff(order).dropna()
    
    def test_adf(self):
        time_series=self.TS.squeeze()       # Convert pd.Series into single list/ array
        test=adfuller(x=time_series,maxlag=self.maxlag,regression=self.regression_type,autolag=self.required_lag,regresults=self.regression_result)
        summary={'test_name':'Augmented Dicky-Fuller Test','ADF_statistics':round(test[0],4),'p_value':round(test[1],4),'used_lag':test[2],'n_observation':test[3],'critical_value':{i:round(j,4) for i,j in test[4].items()},'best_ic':(round(test[5],4) if len(test)>5 else None)}
        return summary
    
    def test_kpss(self):
        time_series=self.TS.squeeze()
        if self.regression_type not in ['c','ct']:
            reg='c'
        else:
            reg=self.regression_type
        test=kpss(x=time_series,regression=reg)
        summary={'test_name':'KPSS Test','KPSS_statistics':round(test[0],4),'p_value':round(test[1],4),'lags':test[2],'critical_value':{i:round(j,4) for i,j in test[3].items()}}
        return summary
    
    def check_stationarity(self):
        """ Combined Decision using:
        ADF: H0 = Non-stationary
        KPSS: H0 = stationary """
        adf_result=self.test_adf()
        kpss_result=self.test_kpss()
        adf_stationarity=(adf_result['p_value']<0.05); kpss_stationarity=(kpss_result['p_value']>0.05)
        stationary=(adf_stationarity and kpss_stationarity)
        summary={'ADF':adf_result,'KPSS':kpss_result,'is_stationary':stationary}
        return summary
    
    def run(self):
        return self.check_stationarity()

if __name__=="__main__":
    local_path = r"C:\Users\HP\OneDrive\Documents\Stock Projects\Vectron\data\storage\HAL.NS_1d_2y.csv"
    if os.path.exists(local_path):
        data_file = pd.read_csv(filepath_or_buffer=local_path)
        series = data_file["Close"]
    else:
        import yfinance as yf
        data_file = yf.download("HAL.NS", period="2y", interval="1d", auto_adjust=True)
        series = data_file["Close"]
    df = pd.DataFrame(data=series)
    df = np.log(df / df.shift(1)).dropna()
    test = StationarityTest(time_series=df, lag_max=10, regression_type="c", required_lag='BIC')
    print(test.run())