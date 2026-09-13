import threading
import datetime, time
import pandas as pd
import numpy as np
# from numba import jit
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.ticktype import TickType, TickTypeEnum
from ibapi.common import TickerId,OrderId
from ibapi.contract import Contract
from IBKR_Connection import ConnectionManager
from collections import defaultdict
from pathlib import Path

class ContractNotFound(Exception):      # Raised when contract can't be qualified with IBKR
    pass

class ContractManager(ConnectionManager):
    Default_Param={'interval':'realtime','period':1,'snapshot':False,'generic_ticks': '100,101,103,104,106,107,165,221,225,233,236,258'}
    def __init__(self):
        ConnectionManager.__init__(self)
        self._lock=threading.Lock()
        self.reqId_to_contract:dict={}      # reqId -> Contract
        self.contract_to_reqId:dict={}      # symbol string -> reqId
        self.callbacks:dict={}
        # Contract Qualification Track
        self.contract_details_response:dict={}      # reqId -> ContractDetails
        self.contract_details_ready:dict={}      # reqId -> threading.event()
        self.reqId_counter=1
    
    def contractDetails(self,reqId:int,contractDetails):
        with self._lock:
            self.contract_details_response[reqId]=contractDetails
        if f"contractDetails_{reqId}" in self.callbacks:
            self.callbacks[f"contractDetails_{reqId}"](contractDetails)
        print(f"Contract is Qualified: {contractDetails.contract.symbol}")

    def contractDetailsEnd(self, reqId:int):        # Callback when contractDetails request end
        if reqId in self.contract_details_ready:
            self.contract_details_ready[reqId].set()
    
    def qualify_contract(self,contract:Contract,timeout:int=5):
        """Validate contracts if they exists in IBKR server
        Args: contract -> Contract object to validate
        Returns: Validate contract object
        Raise: 
         if contract not found: ContractNotFound()
         if IBKR doesn't respond in given time: TimeOutError """
        reqId=self.next_reqID()     # calling method from ConnectionManager()
        self.contract_details_ready[reqId]=threading.Event()
        self.contract_details_response[reqId]=None
        try:
            self.reqContractDetails(reqId=reqId,contract=contract)
            if not self.contract_details_ready[reqId].wait(timeout):
                raise TimeoutError(f"Contract qualification timeout. IBKR didn't responded within {timeout} seconds")
            if reqId not in self.contract_details_response or self.contract_details_response[reqId] is None:
                raise ContractNotFound(f"Contract Not Found: {contract.symbol}|{contract.secType}\nPlease verify symbol, secType and exchange")
            qualified_contract=self.contract_details_response[reqId].contract
            print(f"Successfully Qualified: {qualified_contract.symbol}")
            return qualified_contract
        except Exception as e:
            print(f"Contract Qualification Failed: {str(e)}")
            raise 
        finally:        # cleanup
            self.contract_details_ready.pop(reqId,None)
            self.contract_details_response.pop(reqId,None)
    
    # def contract_constructor(self,symbol_list:list,exchange:list,con_secType:list,con_currency:list):
        
class MarketDataHandler(ContractManager): 
    Default_Param={'interval':'realtime','period':1,'snapshot':False,'generic_ticks': '100,101,103,104,106,107,165,221,225,233,236,258'}
    def __init__(self):
        ContractManager.__init__(self)
        self._lock=threading.Lock()
        # Data Storage
        self.market_data:dict={}; self.historical_data:dict={}
        self.tick_data_series:dict={}
        # Tracking Requests 
        self.reqId_to_contract:dict={}      # reqId -> Contract
        self.contract_to_reqId:dict={}      # symbol string -> reqId
        self.market_data_ready:dict={}
        self.historical_data_ready:dict={}
        # excel export tracking
        self.dataframes:dict={}
        self.callbacks:dict={}
        
    def reqMktData(self,reqId,contract,genericTickList,snapshot,regulatorySnapshot,mktDataOptions):
        EClient.reqMktData(self,reqId=reqId,contract=contract,genericTickList=genericTickList,snapshot=snapshot,regulatorySnapshot=regulatorySnapshot,mktDataOptions=mktDataOptions)

    def cancelMktData(self, reqId):
        try:
            EClient.cancelMktData(self,reqId=reqId)
            with self._lock:
                if reqId in self.market_data_ready:
                    self.market_data_ready[reqId].clear()
            print(f"Market data canceled|reqID:{reqId}")
        except Exception as e:
            print(f"Failed to cancle market data: {str(e)}")
            raise

    def request_market_data(self,contract:Contract,params:dict=None,timeout=5):
        """
    contract: Contract object (after qualification)
    params: {
        'interval': 'realtime'|'5sec'|'10sec' etc,
        'period': 1|5|60|300 seconds or '1 day', '1 week' etc,
        'snapshot': False,  # real-time or snapshot
        'generic_ticks': ''  # optional ticks
    }
    Returns: (reqId, data_collection_event)
    """
        if params is None:
            params=self.Default_Param.copy()
        else: 
            merged_params=self.Default_Param.copy()
            merged_params.update(params)
            params=merged_params
        reqId=self.next_reqID()     # Calling method from ConnectionManager()
        with self._lock:
            self.reqId_to_contract[reqId]=contract
            symbol_key=f"{contract.symbol}_{contract.secType}"
            self.contract_to_reqId[symbol_key]=reqId
            # Initialize data storage for this request
            self.market_data[reqId]={}
            self.tick_data_series[reqId]=defaultdict(list)
            self.market_data_ready[reqId]=threading.Event()
        # Requesting Market data from IBKR
        try: 
            snapshot=params.get('snapshot',False)
            generic_ticks=params.get('generic_ticks',"")
            self.reqMktData(reqId=reqId,contract=contract,genericTickList=generic_ticks,snapshot=snapshot,regulatorySnapshot=False,mktDataOptions=[])
            print(f"Market data requested (reqID={reqId}):{contract.symbol}")
            return reqId
        except Exception as e:
            print(f"Failed to request market data: {str(e)}")
            raise
    
    # =x=x=x=x=x=x=x=x=x=x Requesting Multiple Contracts =x=x=x=x=x=x=x=x=x=x
    def request_multiple_contracts(self,contracts_list:list,params:dict=None):
        """Args:
            contracts_list: List of Contract objects
            params: Optional parameters for all contracts
        Returns: Path to created Excel file"""
        req_ids=[]; successfull_contracts=[];failed_contracts=[]
        for con in contracts_list:
            try:
                qualified=self.qualify_contract(contract=con)
                reqId=self.request_market_data(contract=qualified,params=params)
                req_ids.append(reqId)
                successfull_contracts.append(qualified)
            except Exception as e:
                print(f"Error with {con.symbol}:{str(e)}")
                failed_contracts.append((con.symbol,str(e)))
        if not req_ids:
            raise RuntimeError("Failed to qualify any contract")
        print(f"Successfully qualified {len(successfull_contracts)} contracts")
        if failed_contracts:
            print(f"Failed contracts: {len(failed_contracts)}")
        time.sleep(5)
        all_dataframe={}
        for reqid in req_ids:
            try:
                df=self.get_dataframe(reqid,min_ticks=5,timeout=10)
                if not df.empty:
                    contract=self.reqId_to_contract[reqid]
                    all_dataframe[contract.symbol]=df
            except Exception as e:
                print(f"Failed to collect data for reqID{reqid}: {str(e)} ")
        if all_dataframe:
            excel_path=self.export_to_excel(all_dataframe)
            # cancel all market data requests
            for reqid in req_ids:
                try:
                    self.cancelMktData(reqId=reqid)
                except Exception as e:
                    print(f"Error cancelling market data {reqid}:{str(e)}")
            return excel_path
        else:
            raise RuntimeError("No data collected from any contracts")
    
    # =x=x=x=x=x=x=x=x=x=x=x Data Collection =x=x=x=x=x=x=x=x=x=x  
    def get_dataframe(self,reqId:int,min_ticks:int=10,timeout=30):
        """
        Args: 
            reqId: request ID from request_market_data()
            min_ticks: Minimum ticks to collect before returning
            timeout: seconds to wait for the ticks 
        Returns: pandas.DataFrame
        """
        start_time=time.time()
        while len(self.tick_data_series[reqId].get('bid',[]))<min_ticks:
            if (time.time()-start_time)>timeout:
                print(f"Timeout: Only collected {len(self.tick_data_series[reqId].get('bid',[]))} ticks")
                break
            time.sleep(2)
        with self._lock:        # Converting tick series into DataFrames
            data_dict=dict(self.tick_data_series.get(reqId,{}))
            if not data_dict or 'timestamp' not in data_dict:
                print("No tick data collected")
                return pd.DataFrame()
        df=pd.DataFrame.from_dict(data_dict,orient='index').transpose()
        if 'timestamp' in df.columns:       # Sort data by timestamp
            df['timestamp']=pd.to_datetime(df['timestamp'])
            df=df.sort_values('timestamp',ascending=False)
            df=df.reset_index(drop=True)
        df=df.ffill()       # fill missing values forward
        print(f"DataFrame created: {len(df)} rows x {len(df.columns)} columns")
        with self._lock:
            self.dataframes[reqId]=df
        return df

    def get_historical_dataframe(self,reqId:int,timeout:int=30):
        """Return historical bars collected by historicalData callbacks."""
        ready_event=self.historical_data_ready.get(reqId)
        if ready_event:
            ready_event.wait(timeout=timeout)
        with self._lock:
            bars=list(self.historical_data.get(reqId,[]))
        if not bars:
            print(f"No historical data collected for reqID:{reqId}")
            return pd.DataFrame()
        return pd.DataFrame(bars)
    
    # =x=x=x=x=x=x=x=x=x=x=x Vector Conversion =x=x=x=x=x=x=x=x=x=x  
    def to_vectors(self,dataframe:pd.DataFrame):
        """Args: dataframe: pandas.DataFrame
           Return: dict of {column_name:np.array}"""
        vectors={}
        for column in dataframe.columns:
            try: 
                vectors[column]=np.array(dataframe[column],dtype=np.float64)
            except (ValueError,TypeError):
                vectors[column]=np.array(dataframe[column],dtype=object)
        print(f"converted into {len(vectors)} vector arrays")
        return vectors
    
    # =x=x=x=x=x=x=x=x=x=x=x Excel report  =x=x=x=x=x=x=x=x=x=x  
    def export_to_excel(self,contracts_data:dict=None,filepath:str=None):
        """Args: 
                contract_data: dict of {symbol:dataframe}
                filepath: Output filepath to store the excel file
           Return: Excel file created to path"""
        if contracts_data is None:
            contracts_data={}
            with self._lock:
                for reqId,df in self.dataframes.items():
                    if reqId in self.reqId_to_contract:
                        contract=self.reqId_to_contract[reqId]
                        symbol=contract.symbol
                        contracts_data[symbol]=df
        if not contracts_data:
            raise ValueError("No data to export. Collect data 1st or provide contract_data dict")
        if filepath is None:        # Genereate filepath if not provided
            timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath=f"market_data_{timestamp}.xlsx"
        filepath=Path(filepath)
        filepath.parent.mkdir(parents=True,exist_ok=True)
        try: 
            with pd.ExcelWriter(filepath,engine="openpyxl") as writer:
                for symbol,df in contracts_data.items():
                    if df.empty:
                        print(f"Skipping empty dataframes:{symbol}")
                        continue
                    sheet_name=str(symbol)[:31]     # Sheet name with limit of 31 char
                    if 'timestamp' in df.columns:
                        df_sorted=df.sort_values('timestamp',ascending=False)
                    else:
                        df_sorted=df
                    df_sorted.to_excel(writer,sheet_name=sheet_name,index=False)
                    print(f"Exported sheet: {sheet_name} ({len(df_sorted)} rows)")
            print(f"Excel file created: {filepath}")
            return str(filepath)
        except Exception as e:
             print(f"Failed to export excel:{str(e)}")
             raise
        
    def _tick_str(self, tickType)->str:
        try:
            return TickTypeEnum.toStr(tickType).lower()
        except Exception:
            return str(tickType).lower()

    # =x=x=x=x=x=x=x=x=x=x=x Tick data callback =x=x=x=x=x=x=x=x=x=x   
    def tickPrice(self,reqId:TickerId,tickType:TickType,price:float,attrib):
        with self._lock:
            if reqId not in self.market_data:
                self.market_data[reqId]={}
            if reqId not in self.tick_data_series:
                self.tick_data_series[reqId]=defaultdict(list)
            self.market_data[reqId][tickType]=price     # store both numeric & readable key
            tick_name=self._tick_str(tickType)     # Convert to string
            self.market_data[reqId][tick_name]=price
            timestamp=datetime.datetime.now()       # Add timestamp to series
            self.tick_data_series[reqId]['timestamp'].append(timestamp)
            self.tick_data_series[reqId][tick_name].append(price)
            if tick_name in ("bid","delayed_bid","last","delayed_last","close","delayed_close") and reqId in self.market_data_ready:
                self.market_data_ready[reqId].set()

        if f"tick_{reqId}" in self.callbacks:       # Trigger callback
            self.callbacks[f"tick_{reqId}"](tickType,price)

    def tickSize(self,reqId:TickerId,tickType:TickType,size:int):
        with self._lock:
            if reqId not in self.market_data:
                self.market_data[reqId]={}
            if reqId not in self.tick_data_series:
                self.tick_data_series[reqId]=defaultdict(list)
            tick_name=self._tick_str(tickType)
            self.market_data[reqId][f"{tick_name}_size"]=size
            self.tick_data_series[reqId][f"{tick_name}_size"].append(size)
    
    def tickString(self, reqId, tickType, value):
        with self._lock:
            if reqId not in self.market_data:
                self.market_data[reqId]={}
            if reqId not in self.tick_data_series:
                self.tick_data_series[reqId]=defaultdict(list)
            tick_name=self._tick_str(tickType)
            self.market_data[reqId][f"{tick_name}_str"]=value
            self.tick_data_series[reqId][f"{tick_name}_str"].append(value)

    def historicalData(self,reqId:int,bar):
        with self._lock:
            if reqId not in self.historical_data:
                self.historical_data[reqId]=[]
            self.historical_data[reqId].append({
                "date":bar.date,
                "open":bar.open,
                "high":bar.high,
                "low":bar.low,
                "close":bar.close,
                "volume":bar.volume,
                "bar_count":bar.barCount,
                "wap":bar.wap,
            })

    def historicalDataEnd(self,reqId:int,start:str,end:str):
        print(f"Historical data end for reqID:{reqId}, start:{start}, end:{end}")
        if reqId in self.historical_data_ready:
            self.historical_data_ready[reqId].set()
