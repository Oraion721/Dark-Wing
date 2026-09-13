import time
from ibapi.contract import Contract
from IBKR_MarketData import MarketDataHandler

def build_crypto_contract(symbol,currency,exchange):
    c = Contract()
    c.symbol = symbol
    c.secType = "CRYPTO"
    c.currency = currency
    c.exchange = exchange
    return c

def main():
    client = MarketDataHandler()
    client.connect(host="127.0.0.1", port=7497, clientId=5)  # paper trading
    try:
        client.wait_for_ready(timeout=15)
    except Exception as e:
        print("wait_for_ready failed:", e)
        client.disconnect()
        return

    contract = build_crypto_contract("BTC", "USD", "PAXOS")
    try:
        qualified = client.qualify_contract(contract, timeout=10)
    except Exception as e:
        print("Contract qualification failed:", e)
        client.disconnect()
        return

    req_id = client.next_reqID()
    client.historical_data[req_id] = []
    client.reqHistoricalData(
        reqId=req_id,
        contract=qualified,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=0,
        formatDate=1,
        keepUpToDate=0,
        chartOptions=[]
    )

    time.sleep(10)

    df = client.get_historical_dataframe(req_id)
    print(df.head())

    client.cancelHistoricalData(req_id)
    client.disconnect()

if __name__ == "__main__":
    main()