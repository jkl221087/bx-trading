import time
import requests
import hmac
from hashlib import sha256


APIURL = "https://open-api-vst.bingx.com"
APIKEY = '1VUueCUnGjzMYK4FGNi7wfWKr19I2sjOrcL31nVyNWOSdvYL6WPhVND7CfHWlOSQVEgJ7Ay648nysS04DbsnHQ'
SECRETKEY = 'h8B4G6gVvuz03xNxt9JfxrlQqUbjKX0OFsGsKSms1J1Tw8awuU6aNEYSGHaYUgZpEDG4XGtluOxyVJbyV0UZA'

def get_trading(symbol, side, positionSide, type, quantity):
    payload = {}
    path = "/openApi/swap/v2/trade/order"
    method = "POST"
    paramsMap = {
        "symbol": symbol,
        "side": side,
        "positionSide": positionSide,
        "type": type,
        "quantity": quantity,
    }
    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)

def get_sign(api_secret, payload):
    signature = hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
    return signature

def send_request(method, path, urlpa, payload):
    url = f"{APIURL}{path}?{urlpa}&signature={get_sign(SECRETKEY, urlpa)}"
    headers = {'X-BX-APIKEY': APIKEY}
    response = requests.request(method, url, headers=headers, data=payload)
    return response.text

def praseParam(paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join([f"{x}={paramsMap[x]}" for x in sortedKeys])
    return paramsStr + f"&timestamp={int(time.time() * 1000)}"

if __name__ == '__main__':
    # 在调用 get_trading() 函数时提供所需的参数
    symbol = "ID-USDT"
    side = "BUY"
    positionSide = "LONG"
    type = "MARKET"
    quantity = 10000
    takeProfit = "{\"type\": \"TAKE_PROFIT_MARKET\", \"quantity\": 3,\"stopPrice\": 31968.0,\"price\": 31968.0,\"workingType\":\"MARK_PRICE\"}"
    print("demo:", get_trading(symbol, side, positionSide, type, quantity))