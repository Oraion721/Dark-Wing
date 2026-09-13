from ibapi.client import *
from ibapi.wrapper import *
import time, threading
from ibapi.ticktype import TickTypeEnum
import datetime

# ib_account="DUO958721"
class TestApp(EWrapper,EClient):        # ECLient sends requests to TWS & EWrapper handles incoming messages from TWS
    def __init__(self):
        EClient.__init__(self,self)
        self.account_balance=None
    def nextValidId(self, orderId):
        self.orderId=orderId
    def nextID(self):
        self.orderId+=1
        return self.orderId
    def current_time(self):
        print(time)
    def error(self,reqId,errorCode,errorString):
        print(f"reqID:{reqId}, ErrorCode:{errorCode},ErrorString:{errorString}")
    # EWrapper function
    def updateAccountValue(self, key:str, val:str, currency:str, accountName:str):
        # print(f"Upadate_Account_Value:{key}, Value:{val},Currency:{currency},Account_Name:{accountName}")
        if key=='TotalCashBalance' and currency=='Base':
            print(f"Cash Balance is: {val}")
    def contractDetails(self, reqId, contractDetails):
        attrs=vars(contractDetails)
        print("\n".join(f"{name}:{value}" for name,value in attrs.items()))
        # print(contractDetails.contract)

        myorder=Order()
        myorder.orderId=reqId
        myorder.action="BUY";myorder.orderType="MKT";myorder.totalQuantity=10
        self.placeOrder(reqId,contractDetails.contract,myorder)

        '''myorder.action="BUY";myorder.orderType="LMT";myorder.tif="GTQ";myorder.lmtPrice=144.60
        myorder.totalQuantity=10'''

    def contractDetailsEnd(self, reqId):
        print("End of the contract details")
        self.disconnect()

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"reqId:{reqId},tickType:{TickTypeEnum.to_str(tickType)},price:{price},attrib:{attrib}")

    def tickSize(self, reqId, tickType, size):
        print(f"reqID:{reqId}, ticktype:{TickTypeEnum.to_str(tickType)},size:{size}")
    
    def headTimestamp(self, reqId, headTimestamp):
        print(headTimestamp)
        print(datetime.datetime.fromtimestamp(int(self.headTimestamp)))
        self.cancelHeadTimeStamp(reqId)
    def historicalData(self, reqId, bar):
        print(bar,reqId)
    
    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data end for {reqId}, started at {start}, ending at {end}")
        self.cancelHistoricalData(reqId)
    
    def openOrder(self, orderId, contract, order, orderState):
        print(f"openorder:{orderId},contract:{contract},order:{order}")

    def orderStatus(self, orderId:OrderId, status:str, filled:float, remaining:float, avgFillPrice:float, permId:int, parentId:int, lastFillPrice:float, clientId:int, whyHeld:str, mktCapPrice:float):
        print(f"orderID:{orderId},satus:{status},filled:{filled},remaining:{remaining},avgFillPrice:{avgFillPrice},permID:{permId},parentID:{parentId},lastfillprice:{lastFillPrice},clientID:{clientId},whyHeld:{whyHeld},mktCapPrice:{mktCapPrice}")
        
if __name__ == "__main__":
    application = TestApp()
    application.connect(host="127.0.0.1", port=7497, clientId=0)
    threading.Thread(target=application.run, daemon=True).start()
    time.sleep(2)

    mycontract = Contract()
    mycontract.symbol = "MSFT"
    mycontract.secType = "STK"
    mycontract.exchange = "SMART"
    mycontract.currency = "USD"
    application.reqContractDetails(application.nextID(), mycontract)

    application.reqMarketDataType(3)
    application.reqMktData(reqId=application.nextID(), contract=mycontract, genericTickList="232", snapshot=False, regulatorySnapshot=False, mktDataOptions=[])

    application.reqHeadTimeStamp(reqId=application.nextID(), contract=mycontract, whatToShow='TRADES', useRTH=1, formatDate=1)
    application.reqHistoricalData(reqId=application.nextID(), contract=mycontract, endDateTime="20260407 18:00:00 Asia/Kolkata", durationStr="1D", barSizeSetting="1 hour", whatToShow="TRADES", useRTH=1, formatDate=1, keepUpToDate=True, chartOptions=[])
