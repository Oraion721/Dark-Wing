import threading, time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class ConnectionManager(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self._lock = threading.Lock()
        self.reqID_counter = 1
        self.next_valid_order_ID = None
        self.connected = False
        self.callbacks = {}
        self.data_ready = threading.Event()
        self.api_thread = None

    def next_reqID(self):
        with self._lock:
            req_id = self.reqID_counter
            self.reqID_counter += 1
            return req_id

    def next_orderID(self):
        if self.next_valid_order_ID is None:
            raise RuntimeError("nextValidId not received yet.")
        oid = self.next_valid_order_ID
        self.next_valid_order_ID += 1
        return oid

    def nextValidId(self, orderId):
        self.next_valid_order_ID = orderId
        self.connected = True
        self.data_ready.set()
        print(f"[TWS] nextValidId received: {orderId}")

    def error(self, *args, **kwargs):
        reqId = args[0] if len(args) > 0 else kwargs.get('reqId', -1)
        if len(args) >= 4 and isinstance(args[2], int):
            errorCode = args[2]
            errorString = str(args[3])
        elif len(args) >= 3 and isinstance(args[1], int):
            errorCode = args[1]
            errorString = str(args[2])
        else:
            errorCode = kwargs.get('errorCode', -1)
            errorString = str(kwargs.get('errorString', ''))

        if errorCode in (2104, 2106, 2107, 2108, 2158):
            print(f"[INFO] {errorString}")
        else:
            print(f"[ERROR] reqId={reqId}|code={errorCode}|{errorString}")

    def connectAck(self):
        print("[TWS] Connection acknowledged.")

    def connectionClosed(self):
        self.connected = False
        print("[TWS] Connection closed by TWS.")

    def isConnected(self):
        # Must delegate to EClient.isConnected() so run() loop stays active during handshake
        return super().isConnected()

    def wait_for_ready(self, timeout: int = 10):
        if not self.data_ready.wait(timeout=timeout):
            raise TimeoutError("TWS did not send nextValidId within timeout.")

    def connect(self, host: str = "127.0.0.1", port: int = 4002, clientId: int = 1,
                max_attempts: int = 3, delay_seconds: int = 2):
        if self.connected and self.api_thread and self.api_thread.is_alive() and super().isConnected():
            return

        last_error = None
        for attempt in range(1, max_attempts + 1):
            cid = clientId + (attempt - 1) * 10
            try:
                print(f"[Attempt {attempt}/{max_attempts}] Connecting to {host}:{port} clientId={cid}")
                self.connected = False
                self.data_ready.clear()
                super().connect(host, port, cid)
                self.api_thread = threading.Thread(target=self.run, daemon=True)
                self.api_thread.start()
                if self.data_ready.wait(timeout=10):
                    print(f"[TWS] Connected — attempt {attempt} succeeded with clientId={cid}.")
                    return
                else:
                    print(f"[TWS] Attempt {attempt} timed out waiting for nextValidId.")
                    last_error = "timeout"
                    self.disconnect()
            except Exception as e:
                print(f"[TWS] Attempt {attempt} exception: {e}")
                last_error = e
                self.disconnect()
            if attempt < max_attempts:
                time.sleep(delay_seconds)
        raise RuntimeError(f"Connection failed after {max_attempts} attempts. Last error: {last_error}")

    def disconnect(self):
        try:
            if super().isConnected():
                super().disconnect()
        except Exception as e:
            print(f"[TWS] Disconnect error: {e}")
        self.connected = False
        self.data_ready.clear()
        self.next_valid_order_ID = None
        if self.api_thread and self.api_thread.is_alive() and threading.current_thread() is not self.api_thread:
            self.api_thread.join(timeout=2)
        self.api_thread = None
        print("[TWS] Disconnected.")
