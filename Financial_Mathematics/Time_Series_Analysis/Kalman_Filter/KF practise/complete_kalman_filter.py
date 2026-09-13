import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Optional

# store kalman state each time:
@dataclass
class KalmanState():        # store Kalman state on each timestep
    timestep:int
    beta:float           # State estimate beta(t|t)
    P:float              # Error covariance P(t|t)
    beta_prediction:float      # Predicted state beta(t|t-1)
    P_prediction:float         # Predicted covariance P(t|t-1)
    KalmanGain:float              # Kalman gain K_t
    innovation:float        # residual \epsilon_t=(y_t)-(y_{t|t-1})
    innovation_variance_F_t:float       # Innovation variance F_t=(x_t)^2 * P{t|t-1}+R^2
    log_max_likelihood:float # Contribution to log-likelihood
    spread:float        # spread s_t=y_t-(beta{t|t}*x_t)
    x:float     # independent variable x_t
    y:float     # dependent variable y_t

class KalmanFilter_UniVariate():
    ''' Args:
            Q: Process noise variance(how fast beta changes)
            R: Measurement noise variance(spread volatility)
            initial_beta: Initial hedge ratio guess
            initial_P: Initial uncertainty (high = low confidence)'''
    def __init__(self,Q:float=1e-5,R:float=1e-3,beta_initial:float=1.0,P_initial:float=1000.0):
        self.Q=Q; self.R=R
        self.beta=beta_initial; self.P=P_initial
        self.states:List[KalmanState]=[]        # states will be stored in the list form 
        self.smoothed_beta:List[float]=[]
        self.smothed_P:List[float]=[]
        self.forecast:List[float]=[]
        self.log_likelihood_sum=0
    
    def predict(self)->Tuple[float,float]:
        '''State Prediction (Best prediction will be last beta of the data)
            β̂_{t|t-1}=β̂_{t-1|t-1}
            P_{t|t-1}=P_{t-1|t-1}+Q'''
        beta_predicted=self.beta
        P_predicted=self.P+self.Q
        print(f"beta_pred:{beta_predicted}\nP_pred={P_predicted}")
        return beta_predicted,P_predicted
    
    def correction(self,y:float,x:float,timestep:int=0)->Tuple[float,float,float,float,float]:      # y=dependent price; x=independent price
        r'''
        Args:
            Innovation (residual) \epsilon_t=y_t-(beta{t|t-1}*x_t)
            Innovation var. Var(\epsilon_t)=F_t=x_t^2 * P{t|t-1} + R; R=Var(measurament noise)
            K_t=P_{t|t-1} * X_t / (X_t² * P_{t|t-1} + R)
            State Update: β̂_{t|t}=β̂_{t|t-1} + K_t * (Y_t - β̂_{t|t-1} * X_t)
            Cov. Update: P_{t|t}=(1-K_t*X_t)*P_{t|t-1}
            spread: s_t=y_t-(beta{t|t}*x_t)
        Returns: 
            (beta_updated,spread,KalmanGain,innovation,innovation_var)'''
        if x==0:
            raise ValueError(f"Independent price X_t can not be 0 at timestep {timestep}")
        beta_pred,P_pred=self.predict()
        y_pred=beta_pred*x
        innovation=y-y_pred
        # innovation variance
        F=((x**2)*P_pred)+self.R
        if F<=0:
            raise ValueError(f"Innovation Var. F={F} is not +ve at timestep {timestep}")
        KalmanGain_Gt=P_pred*x/F
        # update the states based on the kalman gain & error
        self.beta=beta_pred+(KalmanGain_Gt*innovation)
        self.P=P_pred*(1-(KalmanGain_Gt*x))
        if self.P<=0:
            print(f"Cov. P={self.P} collapsed at timestep {timestep}")
            print("Resetting Cov. P as initial P")
            self.P=1000
        spread=y-self.beta*x
        log_MLE=-0.5*(np.log(2*np.pi*F)+((innovation**2)/F))
        # store states
        current_state=KalmanState(timestep=timestep,beta=self.beta,P=self.P,beta_prediction=beta_pred,P_prediction=P_pred,KalmanGain=KalmanGain_Gt,innovation=innovation,innovation_variance_F_t=F,log_max_likelihood=log_MLE,spread=spread,x=x,y=y)
        self.states.append(current_state)
        print(f"[timestep={timestep}]|Correction beta={self.beta:.6f}|Cov. P={self.P:.6f}|spread={spread:.6f}|innovation epsilon_t={innovation:.6f}")
        return self.beta,spread,KalmanGain_Gt,innovation,F
    
    def Forecast_Multistep(self,future_x:np.ndarray,steps_ahead:int=1,curent_timestep:int=0)->Tuple [np.ndarray,np.ndarray,np.ndarray]:
        '''
        Equations:
            β̂_{t+h|t} = β̂_{t|t}  (β doesn't change in prediction)
            Ŷ_{t+h|t} = β̂_{t|t} * X_{t+h}
            P_{t+h|t} = P_{t|t} + h * Q  (uncertainty grows)
            Var(Ŷ_{t+h|t}) = X_{t+h}² * P_{t+h|t} + R
            Spread forecast: ŝ_{t+h|t} = Ŷ_{t+h|t} - β̂_{t|t} * X_{t+h}
        Args:
                future_x: Future independent variable values [X_{t+1}, X_{t+2}, ...]
                steps_ahead: Number of steps to forecast
        Returns: 
            (y_forecast,spread_forecast,forecast_var)'''
        if len(self.states)==0:
            raise ValueError ("Must run correction() function before running ForecastMultistep()")
        current_state=self.states[-1]
        current_beta=current_state.beta
        current_P=current_state.P
        y_forecast=[];forecast_variance=[]
        spread_forecast=[]
        for h in range(1,steps_ahead+1):
            if h<=len(future_x):
                x_h=future_x[h-1]
            else:
                x_h=future_x[-1] if len(future_x)>0 else 1
            y_h=current_beta*x_h        # y_{t+h|t}=beta_{t|t}*X_{t+h}
            P_h=current_P+h*self.Q
            var_y_h=((x_h**2)*P_h)+self.R
            spread_h=y_h-(current_beta*x_h)
            y_forecast.append(y_h)
            spread_forecast.append(spread_h)
            forecast_variance.append(var_y_h)
        return np.array(y_forecast),np.array(spread_forecast),np.array(forecast_variance)
    
    # def forecast_spread(self,x_current:float,y_current:float,x_future:np.ndarray,step_ahead:int=5):
        """Spread_{t+h} = Y_{t+h} - β̂_{t|t} * X_{t+h}"""
        if self.states==0:
            raise ValueError("Must run MultiStepAheadForcasting() function before running forecast_spead()")
        y_forecast,y_var=self.MultiStepAheadForecasting(future_x=x_future,steps_ahead=step_ahead)
        current_state=self.states[-1]
        expected_spread=[]
        for h, (y_pred, x_h) in enumerate(zip(y_forecast, x_future[:step_ahead])):
            spread_pred=y_pred-current_state.beta*x_h
            spread_var=y_var[h]+x_h**2*current_state.P
            expected_spread.append(spread_pred)
        return np.array(expected_spread)
    
    def Smoothing(self)-> Tuple[List[float],List[float]]:
        """Use smoothing matrix S_t to refine the past state estimates using future info. 
        Equations:
            t'=t-1,t-2,t-3,... (past time)
            t=current time 
            Kalman Smoothing Matrix (Smoothing Gain) S_t'=P{t'|t'}/P{t'+1|t'}
            beta_{t'|t}=beta{t|t}+[(S_t')*(beta{t'+1|t}-beta{t'+1|t'})]
            P{t'|t}=P{t'|t'}-(S_t')*(P{t'+1|t'})+(S_t')*(P{t'+1|t}) 
        Returns:
         (smoothed beta, smoothed P):Smoothed state estimates """
        if len(self.states)<2:
            print("Not enough states for smoothing (need>2)")
            return [],[]
        t=len(self.states)
        smoothed_beta=[0]*t
        smoothed_P=[0]*t
        smoothed_beta[-1]=self.states[-1].beta
        smoothed_P[-1]=self.states[-1].P
        for T in range(t-2,-1,-1):
            beta_t=self.states[-1].beta
            P_t=self.states[-1].P
            beta_t1_pred=self.states[-1].beta_prediction
            P_t1_pred=self.states[-1].P_prediction
            # Smoothing gain
            if P_t1_pred>0:
                S_t=P_t/P_t1_pred
            else:
                S_t=0
            smoothed_beta[t]=beta_t+S_t*(smoothed_beta[t+1]-beta_t1_pred)
            smoothed_P[t]=P_t+S_t*(smoothed_P[t+1]-P_t1_pred)*S_t
        self.smoothed_beta=smoothed_beta
        self.smoothed_P=smoothed_P
        return smoothed_beta,smoothed_P

    def Log_MLE(self):
        r"""
        Equation:
            log(L_t)=-0.5[log(2*3.14*F_t)+epsilon_t^2/F-t]
            total log likelihood: log(L)=\sum (t=1) to T{log(L_t)}
        Return:
            Total log likelihood sum    """
        return self.log_likelihood_sum
    
    def optimize_parameter(self,y_data:np.ndarray,x_data:np.ndarray,Q_bounds:Tuple[float,float]=(1e-6,1e-2),R_bounds:Tuple[float,float]=(1e-4,1.0))->dict:
        """Estimate optimal Q & R using MLE, minimize -ve log-likelihood over historical data
        Args:
            y_data: Dependent price array
            x_data: Independent price array
            Q_bounds: Search bounds for Q
            R_bounds: Search bounds for R
        Returns:
            Dictionary with optimal parameters and convergence info"""
        temp_filter=KalmanFilter_UniVariate()
        def neg_log_likelihood(params):
            temp_filter.reset()
            temp_filter.Q=params[0]
            temp_filter.R=params[1]
            for y,x in zip(y_data,x_data):
                try: 
                    temp_filter.correction(y,x)
                except:
                    return 1e10
            return -temp_filter.Log_MLE()
        initial_guess=[self.Q,self.R]
        bounds=[Q_bounds,R_bounds]
        result=minimize(neg_log_likelihood,initial_guess,method="L-BFGS-B",bounds=bounds,options={'maxiter':100})
        if result.success:
            self.Q,self.R=result.x
            print(f"Parameter Optimized: Q={self.Q:.6f}|R={self.R:.6f}")
        else:
            print(f"Warning: Optimization did not converge: {result.message}")
        return {'Q_optimal':result.x[0],'R_optimal':result.x[1],'log_likelihood':-result.fun,'success':result.success,'n_iteration':result.nit}

    def generate_trading_signal(self,y:float,x:float,entry_threshold:float=2.0,exit_threshold:float=0.5,forecast_steps:int=5,timestep:int=0)->dict:
        '''
        Signal logic: 
        If spread>(entry_threshold*std_dev): signal = -1 (short sperad)
        If spread<(-entry_threshold*std_dev): signal = 1 (long sperad)
        If |spread|<(exit_threshold*std_dev): signal = 0 (exit)
        Else: sigal=0 (hold)
        Args:
            y: Current Y price
            x: Current X price
            entry_threshold: Number of standard deviations to enter
            exit_threshold: Number of standard deviations to exit 
            forecast_step: Steps ahead to forecast for reversion time
            timestep: Current timestep
        Returns:
            signal: -1 (short spread), 0 (neutral), 1 (long spread)
            spread: Current spread value
            info: Additional forecasting info
            beta, reversion time'''
        beta,spread,KalmanGain_Gt,innovation,F=self.correction(y,x,timestep)
        current_state=self.states[-1]
        spread_std=np.sqrt(current_state.innovation_variance_F_t)
        # forecast next 5 period for exit
        future_x=np.array([x*(1+0.0001*i) for i in range(1,forecast_steps+1)])
        _,spread_forecast,_=self.Forecast_Multistep(future_x=future_x,steps_ahead=forecast_steps,curent_timestep=timestep)
        reversion_time=self.estimate_mean_reversion_time(spread_forecast)
        # signal generation
        if spread>entry_threshold*spread_std:   # high threshold
            signal=-1       # Sell y & buy x
        elif spread<-entry_threshold*spread_std:
            signal=1        # Sell x, buy y
        elif abs(spread)<exit_threshold*spread_std:
            signal=0        # exit position
        else:
            signal=0        # hold the position
        info={'beta':beta,'spread':spread,'KalmanGain':current_state.KalmanGain,'innovation':innovation,'signal':signal}
        return signal,spread,info
    @staticmethod
    def estimate_mean_reversion_time(future_spread:np.ndarray):     # Estimate how many steps ahead to mean revert
        if len(future_spread)==0:
            return None
        zero_crossings = np.where(np.diff(np.sign(future_spread)))[0]
        if len(zero_crossings)>0:
            return zero_crossings[0]+1
        return None
    
    def reset(self):        #Reset filter for new pair or backtest
        self.beta=1.0
        self.P=1000.0
        self.states=[]
        self.smoothed_beta=[]
        self.forecast=[]
        self.smoothed_P=[]
        self.log_likelihood_sum=[]
