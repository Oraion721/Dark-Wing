import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

class ADF_Test:
    def __init__(self,price_series:pd.DataFrame,max_lags:int,variant:int):
        """Class inputs required to compute the Augamented Dicky-Fuller Test
        Args:
            ~ price_series:pd.DataFrame: Series or DataFrame of asset price
            ~ max_lags:int: Lags required to perform Akaike Information Criterian(AIC) and NG-Perron max_lag
            ~ variants:int: Used to build the regrresor matrix [X]
                0 = no constant, no trend
                1 = constant only
                2 = constant and trend """
        self.price_series=price_series
        self.max_lags=max_lags
        self.variant=variant
        self.prices=self.price_series.iloc[:,0].values       # convert prices into 1-D array assuming 1st clm is prices
    
    def LR(self):
        self.log_returns=np.log(self.prices)        # convert the prices into log 
        self.diff_log_returns=np.diff(self.log_returns)      # First differences: Δp_t = p_t - p_{t-1} 
    
    def regressor_matrix(self):
        """ Build the response matrix y and regressor matrix X using the same sample
        y=[Δp_t] from t=(p+2,p+3,p+4,....T), p=number of lags
        X=[1,P_{t-1},ΔP_{t-1},ΔP_{t-2},ΔP_{t-3},.....ΔP_{t-p}]        """
        T=len(self.log_returns)
        start=self.max_lags     # 1st index for y
        end=T-1     # last index for y
        self.n_observation=T-self.max_lags-1        # no. of usable observation
        y=self.diff_log_returns[start:end]
        lagged_level=self.log_returns[self.max_lags:T-1]        # T=len(self.log_returns)
        trend=np.arange(self.max_lags+1,T)
        lagged_diff=[]      # empty list of lagged series
        for i in range(1,self.max_lags+1):
            lag_diff=self.diff_log_returns[self.max_lags-i:T-1-i]
            lagged_diff.append(lag_diff)
        self.lagged_diff=np.column_stack(lagged_diff) if lagged_diff else np.empty((self.n_observation,0))
        # storing components
        self.y=y
        self.lagged_level=lagged_level
        self.trend=trend

    def ols_regression(self,X,y):
        X_transpose_X=X.T@X     # X'X
        X_transpose_X_inverse=np.linalg.pinv(X_transpose_X)     # (X'X)^-1
        theta=X_transpose_X_inverse@(X.T@y)     # theta=((X'X)^-1)*(X'y)
        residual=y-(X@theta)
        n=len(y)
        sigma_square=(residual@residual)/n
        return theta,residual,sigma_square,X_transpose_X_inverse
    
    def compute_AIC(self,sigma2,n_parameters,n_obervation):
        AIC_k=np.log(sigma2)+(2*n_parameters)/n_obervation
        return AIC_k
    
    def build_X_for_lags(self,k):
        cols=[]
        if self.variant>=1:
            cols.append(np.ones(self.n_observation))
        cols.append(self.lagged_level)
        if self.variant==2:
            cols.append(self.trend)
        if k>0:
            cols.append(self.lagged_diff[:,:k])
        X=np.column_stack(cols)
        return X
    
    def select_optimal_lags(self):
        """Loop over k=(0,1,2,....max_lag), compute AIC, return best lag"""
        best_AIC=np.inf
        best_k=0
        for k in range(0,self.max_lags+1):
            X=self.build_X_for_lags(k)
            theta,residual,sigma_square,X_tran_X_inv=self.ols_regression(X,self.y)
            n_parameters=1
            if self.variant>=1:
                n_parameters+=1
            if self.variant==2:
                n_parameters+=1
            n_parameters+=k
            AIC=self.compute_AIC(sigma2=sigma_square,n_parameters=n_parameters,n_obervation=self.n_observation)
            if AIC<best_AIC:
                best_AIC=AIC
                best_k=k
                self.best_theta=theta; self.best_X=X
                self.best_sigma_square=sigma_square; self.best_X_T_X_inv=self.build_X_for_lags(k=k)
        self.best_lag=best_k
        self.best_AIC=best_AIC
        return best_k
    
    def tau_statistics(self):
        """tau=delta/SE(delta)
        Standard Error SE(delta): sqrt((best_sigma**2)*((X'X)^-1)11)
        Residual=y-(best_X@best_theta)
        In X: order: constant (if any), lagged_level, trend (if any), then lags"""
        if self.variant>=1:
            delta_position=1        # 2nd clm
        if self.variant==2:
            delta_position=1        # 1st clm
        delta_hat=self.best_theta[delta_position]
        n=self.n_observation; p=len(self.best_theta)
        residual=self.y-(self.best_X@self.best_theta)
        sigma_unbiased=(residual@residual)/(n-p)
        XT_X_inv=np.linalg.pinv(self.best_X.T@self.best_X)
        SE_delta=np.sqrt((sigma_unbiased*XT_X_inv[delta_position,delta_position]))
        tau=delta_hat/SE_delta
        return tau
    
    def critical_value(self):
        """Return critical values for 1%,5%,10% based on variant."""
        if self.variant==0:          # no constant, no trend
            return {-1:-2.58,-2:-1.95,-3:-1.62}   # using negative for thresholds
        elif self.variant==1:        # constant only
            return {-1:-3.43,-2:-2.86,-3:-2.57}
        else:                          # constant and trend
            return {-1:-3.96,-2:-3.41,-3:-3.13}
    
    def p_value_approximation(self, tau):
        """Approximate MacKinnon p-value using regression"""
        if self.variant == 1:  # constant only
            # MacKinnon coefficients for constant case
            cv_points=[-3.43,-2.86,-2.57]
            p_points=[0,0.001,0.01,0.05,0.10]
            if tau<cv_points[0]:
                p=p_points[1]
            elif tau<cv_points[1]:
                p=p_points[2]
            elif tau<cv_points[-1]:
                p=p_points[3]
            elif tau>-1:
                p=p_points[0]
            else:
                p=p_points[-1]
        return p
    
    def run_adf(self):
        self.LR()
        self.regressor_matrix()
        best_lag=self.select_optimal_lags()
        tau=self.tau_statistics()
        critical_values=self.critical_value()
        result_str={}
        for sig,civ in critical_values.items():
            sig_name={-1:'1%',-2:'5%',-3:'10%'}[sig]
            if tau<civ:     # more -ve: reject unit root
                result_str[sig_name]="Reject"
            else:
                result_str[sig_name]="Not Rejected"
        output={"tau statistics":tau,"optimal_lag":best_lag,"variant":self.variant,"critical values":critical_values,"result":result_str,"n_observation":self.n_observation,"best_AIC":self.best_AIC}
        return output
    
    def summary(self):
        res=self.run_adf()
        print("="*20+"ADF Test Result"+"="*20)
        print(f"variant={res['variant']}",end="")
        if res['variant']==0:
            print('No Constant, No Trend')
        elif res['variant']==1:
            print('Constant Only')
        else:
            print("Constant & Trend")
        print(f"Optimal numbers of lags(by AIC):{res['optimal_lag']}")
        print(f"tau statistics (delta/SE(delta): {res['tau statistics']:.4f}")
        pval=self.p_value_approximation(tau=res['tau statistics'])
        print(f"P-value:{pval:.4f}")
        print(f"Observations:{res['n_observation']}")
        print(f"AIC:{res['best_AIC']:.4f}")
        print("\nCritical Values")
        for sig,cv in res['critical values'].items():
            sig_name={-1:'1%',-2:'5%',-3:'10%'}[sig]
            print(f"{sig_name}:{cv:.2f}")
        for sig,decision in res['result'].items():
            print(f"At {sig} significance level:{decision}")
        return res

if __name__ == "__main__":
    asset = "BSE.NS"
    start = "2016-01-01"
    end = "2026-05-25"
    data_file = yf.download(tickers=asset, start=start, end=end, auto_adjust=True, interval="1d")
    df = pd.DataFrame(data=data_file["Close"])
    print(df)
    # Suppose you have a DataFrame 'df' with a column 'price'
    adf = ADF_Test(df, max_lags=10, variant=1)
    results = adf.summary()
    result = adfuller(df, maxlag=10, autolag='AIC', regression='c')
    print(result)