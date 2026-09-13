import numpy as np; import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from statsmodels.tsa.api import VAR

""" Usage:
        from Optimal_lag_selection import VARLagSelector
        selector = VARLagSelector(data=df, maxlags=10, criteria="aic")
        results  = selector.select_lag()
        selector.summary()
        selector.plot() """
_IC_color={'aic':'blue','bic':'yellow','hqic':'green','fpe':'red'}
_valid_criteria=['aic','bic','hqic','fpe']
_valid_transform=['none','log','returns']
_valid_trend=['n','c','ct','ctt']

class VAROptimalLagSelect:      # VAR optimal lag (p*) order selection using VAR.select_order()
    def __init__(self,data:pd.DataFrame,maxlags:int=None,criteria:str="aic",transform:str='none',trend:str="c",):
        """Args:
            ~ data:pd.DataFrame: Data containing m TS(log prices)
            ~ maxlags:int: Max number of lags; if None than use Ng-Perron formula 12*(n_observations/100)^0.25
            ~ criteria:str: default "aic"
                Information criterion to minimise. Must be one of: ["aic", "bic", "hqic", "fpe"]
            ~ transform:str="none": given data is which type? 
                    transform should be ['none','log','returns']
                    'none': if the provided data is already "log" prices
                    'log': if the provided data is raw prices and need to convert in "log" prices
                    'returns': if the provided data is raw prices and need to convert in "log(P_t - P_(t-1))"
            ~ trend:str="c": Deterministic trend for the VAR. Must one of ['n','c','ct','ctt']
        O/P:
            ~ summary and plot of the computations """
        
        self.data=data.copy()
        self.criteria=criteria.lower().strip()      # str.strip() -> removes any remaining space after last char
        self.transform=transform.lower().strip()
        self.trend=trend.lower().strip()
        self._transform_data()
        self.optimal_lag:int=None
        self.ic_table:pd.DataFrame=None
        self.σ_μ:np.ndarray=None        # residual cov at p*
        self.log_prices=None
        self._fitted_model=None     # VAR(p*) fitted model

        # ======================== Validate Args: ===========================
        if self.criteria not in _valid_criteria:
            raise ValueError(f'Criteria must be one of {_valid_criteria}, but got {self.criteria}')
        if self.transform not in _valid_transform:
            raise ValueError(f"Transform must be one of {_valid_transform}, but got {self.transform}")
        if self.trend not in _valid_trend:
            raise ValueError(f"Tend must be one of {_valid_trend}, but got {self.trend}")
        
        # ======================== Validate Dataframe ==========================
        if not isinstance(self.data,pd.DataFrame):
            raise TypeError("Data must be pd.DataFrame")
        if self.data.empty:
            raise ValueError("Data is empty")
        if self.data.shape[1]<2:
            raise ValueError(f"VAR requires >=2 TS, got {self.data.shape[1]}")
        
        # =========== Apply Transformation and Resolve maximum lags =============
        self._prepared=self._transform_data()       # Transform data depending upon transform Arg
        self.T=len(self._prepared)      # len of total observations
        self.m=self._prepared.shape[1]      # no. of series

        if maxlags is None:
            self.maxlags=self._auto_maxlag(self.T)
        else:
            if not isinstance(maxlags,int) or maxlags<1:
                raise ValueError("maxlags must be > 1")
            if maxlags>=(self.T-10):
                raise ValueError(f"Maxlags ({maxlags}) is too large for T={self.T}\nMaxlag must be < {self.T-10}")
            self.maxlags=maxlags
        
    def __repr__(self):
        status=(f"Optimal_lag={self.optimal_lag}" if self.optimal_lag is not None else "not fitted yet - call select_lag()")
        return (f"{"="*50}\n"
                f"VAROptimalLagSelector("
                f"m={self.m}, T={self.T},"
                f"maxlags={self.maxlags},"
                f"criteria={self.criteria.upper()},"
                f"{status})")
    
    def _transform_data(self):      # Extract log prices:by default we assume that the input already contains log prices
        df=self.data.copy()
        if self.transform=='log':
            if (df<=0).any().any():
                raise ValueError("transform='log': All Dataframe values should be > 0.\nFound non-positive values in data.")
            df=np.log(df)
        elif self.transform=='returns':
            if (df<=0).any().any():
                raise ValueError("transform='returns': All Dataframe values should be > 0.\nFound non-positive values in data.")
            df=np.log(df/df.shift(1))
        df=df.replace([np.inf,-np.inf],np.nan).dropna()
        if df.empty:
            raise ValueError('Data is Empty after removal of NaN values')
        
        return df 

    @staticmethod    
    def _auto_maxlag(T:int):
        return max(int(np.floor(12*(T/100)**0.25)),1)       # Ng-Perron Formula for getting maxinmum lags
    
    def _require_fitted(self,method_name:str):
        if self.ic_table is None:
            raise RuntimeError(f"Call select_lag() before calling {method_name}().")

    # ==================== Main Method =========================
    def select_lag(self):       # perform VAR order selection using statsmodels
        """Lag order selection:
        statsmodels.tsa.api.VAR.select_orders(maxlags,trend)
        Args:
            ~ maxlags:int: Max. lags allowed to test
            ~ trend:str:"c"
                "c"=constant only; "ct"=constant+linear; "n"=None
        O/P:
            dict with keys {
                "optimal_lag":int      - best lag p* for chosen criteria
                "criteria":str         - which IC was used
                "all_criteria":dict    - dict: {'aic':int,'bic':int,...}
                "ic_table":pd.DataFrame - IC scores of each lag
                "sigma_u":np.ndarray } """
        
        log_prices=self._prepared
        model=VAR(log_prices)       # initiating stasmodels.tsa.api.VAR

        try:
            lag_selection=model.select_order(maxlags=self.maxlags,trend=self.trend)   
        except Exception as e:
            raise RuntimeWarning(f"VAR.select_order() failed: {e}") from e
        
        aic_length=len(lag_selection.ics["aic"])
        if aic_length==0:       # check if ics are populated
            raise ValueError("lag_selection.ics['aic'] is empty, check data and maxlag values")
        
        # ================ Extract IC_table =================
        self.ic_table=pd.DataFrame({crit:lag_selection.ics[crit] for crit in _valid_criteria},index=range(0,aic_length))
        self.ic_table.index.name='lag'

        # ============ Extract p* for each criteriaon ====================
        all_optimal={crit:int(lag_selection.selected_orders[crit]) for crit in _valid_criteria}
        self.optimal_lag=all_optimal[self.criteria]

        # =========== Fit VAR(p*) to get residual cov matrix =============
        if self.optimal_lag==0:
            warnings.warn(f"\noptimal lag=0 selected by {self.criteria.upper()}\nThis means no temporal dependence was found, Setting lag=1 for covariance estimation",UserWarning)
            fit_lag=1
        else:
            fit_lag=self.optimal_lag
        self._fitted_model=model.fit(maxlags=fit_lag,trend=self.trend)
        self.σ_μ=self._fitted_model.sigma_u

        return {'optimal_lag':self.optimal_lag,'criteria':self.criteria,'all_criteria':all_optimal,'ic_table':self.ic_table,"σ_μ":self.σ_μ}
    
    def summary(self):
        self._require_fitted("summary")
        sep = "=" * 62
        print(f"\n{sep}")
        print("  VAR LAG ORDER SELECTION SUMMARY")
        print(sep)
        print(f"  Series (m)      : {self.m}")
        print(f"  Observations (T): {self.T}")
        print(f"  Max lags tested : {self.maxlags}")
        print(f"  Trend           : {self.trend.upper()}")
        print(f"  Selection basis : {self.criteria.upper()}")
        print(f"{sep}")
        if self.ic_table is None:
            print("Run select_lag method 1st!")
            return
        ic_display=self.ic_table.copy()
        ic_display.columns=[c.upper() for c in ic_display.columns]
        marker_col=self.criteria.upper()
        opt_val=ic_display.loc[self.optimal_lag,marker_col]
        ic_display[marker_col]=ic_display[marker_col].apply(lambda v:f">>>{v:.6f}<<<" if v==opt_val else f"     {v:.6f}")
        for col in ic_display.columns:
            if col!=marker_col:
                ic_display[col]=ic_display[col].apply(lambda v:f"{v:.6f}")
        print(ic_display.to_string())
        print(sep)
        print(f"    Optimal_lag p*={self.optimal_lag}\n[{self.criteria.upper()} minimised]")
        print(sep)
        print('All criteria decisions:')
        for crit in _valid_criteria:
            p=self.ic_table[crit].idxmin()
            flag="  ← selected" if crit==self.criteria else ""
            print(f"    {crit.upper():5s} → p*={p}{flag}")
        print(f'{sep}\n')

    def plot(self,figsize:tuple=(12,5)):
        self._require_fitted('plot')
        fig,axes=plt.subplots(1,4,figsize=figsize)
        fig.suptitle(f'VAR Lag Selection - p* = {self.optimal_lag}\n[{self.criteria.upper()}]',fontsize=13,fontweight='bold',y=1.02)

        for ax,(crit,color) in zip(axes,_IC_color.items()):
            values=self.ic_table[crit]
            opt_lag=values.idxmin() # if crit==self.criteria else values.idxmin()
            ax.plot(values.index,values.values,marker="o",color=color,linewidth=1.8,markersize=5)
            ax.axvline(opt_lag,color="red",linestyle="--",linewidth=1.2)
            if crit==self.criteria:
                ax.axvline(opt_lag,color='red',linestyle="--",linewidth=2.0,label=f"p* = {opt_lag}")
                ax.scatter([opt_lag],[values[opt_lag]],color="red",zorder=5,s=90,label=f"p*={opt_lag}")
            ax.set_title(f"{crit.upper()}",fontweight="bold",color=color)
            ax.set_xlabel("Lag order p", fontsize=9)
            ax.set_ylabel("IC value",fontsize=9)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_optimal_lag(self):
        self._require_fitted('get_optimal_lag')
        return self.optimal_lag

    def get_σ_μ(self):
        self._require_fitted("get_σ_μ")
        return self.σ_μ
    
if __name__=="__main__":
    data=yf.download(tickers=['BSE.NS','BEL.NS','SBIN.NS'],period='4y',interval='1d',auto_adjust=True)['Close']
    df=pd.DataFrame(data=data)
    selector=VAROptimalLagSelect(data=df,criteria="aic",transform='log',trend='c')
    results=selector.select_lag()
    selector.summary()
    selector.plot()