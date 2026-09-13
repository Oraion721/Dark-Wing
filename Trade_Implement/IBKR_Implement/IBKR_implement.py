import time
from ibapi.contract import Contract
from IBKR_MarketData import MarketDataHandler

def build_fx_contract(symbol="BTC",currency="USD",exchange="PAXOS"):
    c=Contract()
    c.symbol=symbol; c.secType="CRYPTO"
    c.currency=currency
    c.exchange=exchange
    return c

def main():
    client=MarketDataHandler()
    # connect to TWS
    client.connect(host="127.0.0.1",port=7497,clientId=5)
    # client.wait_for_ready(timeout=10)
    print(f"Connection Status:{client.isConnected()}")
    try:
        client.wait_for_ready(timeout=15)
    except Exception as e:
        print(f"wait for ready failed:{e}")
        print(f"next_valid_order_ID:{client.next_valid_order_ID}")
        client.disconnect()
        return
    print(f"Ready! NextValidID:{client.next_valid_order_ID}")
    # qualify fx contract
    fx_contract=build_fx_contract("BTC","USD","PAXOS")
    try: 
        qualified=client.qualify_contract(fx_contract,timeout=10)
        print(f"Qualified contract: {qualified}")
    except Exception as e:
        print("contract qualification failed:",e)
        client.disconnect()
        return
    
    # request market data:
    req_id=client.request_market_data(contract=qualified,params={'snapshot':False,'generic_ticks':"100,101,104"})
    print(f"Requested market data reqId:{req_id}")
    time.sleep(10)
    df=client.get_dataframe(reqId=req_id,min_ticks=3,timeout=20)
    print(df.head())
    client.cancelMktData(req_id)
    client.disconnect()

if __name__=="__main__":
    main()
