import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import csv
from market import Market
from trade import Trade
from bingX._http_manager import _HTTPManager
from typing import Any
import time
from trdingget import get_trading
import sys
import os
import plotly.graph_objects as go
from binance.client import Client
from datetime import datetime
sys.path.append(os.path.abspath("../"))
from smartmoneyconcepts.smc import smc
import json

# 設定api鑰駛
api_key = "1VUueCUnGjzMYK4FGNi7wfWKr19I2sjOrcL31nVyNWOSdvYL6WPhVND7CfHWlOSQVEgJ7Ay648nysS04DbsnHQ"
secret_key = "h8B4G6gVvuz03xNxt9JfxrlQqUbjKX0OFsGsKSms1J1Tw8awuU6aNEYSGHaYUgZpEDG4XGtluOxyVJbyV0UZA"


class TradingBot:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.client = Client(api_key, secret_key)
        self.position = 0

    def get_latest_price(self, symbol="BTCUSDT"):
        latest_prices_data = self.client.get_symbol_ticker(symbol=symbol)
        latest_price = float(latest_prices_data['price']) if 'price' in latest_prices_data else None
        return latest_price
    
    def import_data (self, symbol, start_str, end_str, timeframe):
        df = pd.DataFrame(
            self.client.get_historical_klines(
                symbol=symbol, interval=timeframe, start_str=start_str, end_str=end_str
            )
        ).astype(float)
        df = df.iloc[:,:6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit = "ms").strftime("%Y-%m-%d %H:%M:%S")
        return df
    
    def calculate_ema(self, prices, span):
        """這行程式碼使用了 Pandas 庫中的指數加權移動平均
        （Exponential Weighted Moving Average，EMA）函數。
        具體而言，ewm 方法用於計算指定窗口大小的指數加權移動平均值，而 mean() 
        方法用於計算這些指數加權移動平均值的均值。
        adjust=False 參數表示不使用調整係數進行平滑，而是使用固定的權重。
        mean() 方法用於計算指數加權移動平均值的均值。"""
        prices_series = pd.Series(prices)
        ema = prices_series.ewm(span = span, adjust=False).mean() 
        return ema
    
    #計算背離
    def calculate_divergence(self, prices, indicator):
        """
        计算价格与指标之间的背离

        参数：
        - prices: 包含价格数据的列表或数组
        - indicator: 包含指标数据的列表或数组，长度与prices相同

        返回值：
        - divergence: 背离的类型，"顶部背离"、"底部背离" 或 "无背离"
        """
        if len(prices) != len(indicator):
            raise ValueError("Prices and indicator lengths must be equal.")

        # 计算价格和指标的变化
        price_changes = np.diff(prices)
        indicator_changes = np.diff(indicator)

        # 判断最后一个变化的方向
        price_change_sign = np.sign(price_changes[-1])
        indicator_change_sign = np.sign(indicator_changes[-1])

        # 判断背离类型
        if price_change_sign == 1 and indicator_change_sign == -1:
            divergence = "頂背離"
        elif price_change_sign == -1 and indicator_change_sign == 1:
            divergence = "底背離"
        else:
            divergence = "無背離"

        return divergence
    
    def generate_signals(self, divergence):
        if divergence == "頂背離":
            signal = 'SELL'
        elif divergence == "底背離":
            signal = 'BUY'
        else:
            signal = "Hold"
        return signal
    

    def signal_trading(self, signal, latest_price):
            # Your trading logic here
            """計算止盈止損價格"""
            #止損
            stop_loss_price = latest_price // 0.98
            print("止損", stop_loss_price)
            #止盈
            take_profit_price = latest_price * 9
            print("止盈", take_profit_price)

            take_profit_payload = f'{{"type": "TAKE_PROFIT_MARKET", "stopPrice": {take_profit_price}, "price": {take_profit_price}, "workingType": "MARK_PRICE"}}'
            stop_loss_payload = f'{{"type": "STOP_MARKET", "stopPrice": {stop_loss_price}, "price": {stop_loss_price}, "workingType": "MARK_PRICE"}}'

            if self.position != 0:
                print("Already have position, no trading needed.")
                return self.position
            
            if signal == 'BUY':
                # 在买入时设置止盈价格和止损价格
                trading_response = get_trading(
                    symbol="BTC-USDT",
                    side="BUY",
                    positionSide="LONG",
                    type="MARKET",
                    quantity=1,  # 这里根据您的需求设置下单数量
                    takeProfit=take_profit_payload,
                    stopLoss=stop_loss_payload
                )
                print("Trading response:", trading_response)
                position = 1  # 更新仓位为1，表示持有多头仓位
            elif signal == 'SELL':
                # 在卖出时设置止盈价格和止损价格
                trading_response = get_trading(
                    symbol="BTC-USDT",
                    side="SELL",
                    positionSide="SHORT",
                    type="MARKET",
                    quantity=1,  # 这里根据您的需求设置下单数量
                    takeProfit=take_profit_payload,
                    stopLoss=stop_loss_payload
                )
                print("Trading response:", trading_response)
                position = -1  # 更新仓位为-1，表示持有空头仓位
            else:
                print("No trading signal generated.")

            return self.position
    

    def run_strategy(self, symbol, start_date, end_date, timeframe):
        running = True
        while running:
            latest_price = self.get_latest_price()
            print("最信價格:", latest_price)
            if latest_price is not None:
                df = self.import_data(symbol, start_date, end_date, timeframe)
                prices = df['close'].tolist() # .tolist() 方法將該列中的數據轉換為 Python 列表
                ema_20 = self.calculate_ema(prices, span=20)
                divergence = self.calculate_divergence(prices, ema_20)
                print("趨勢", divergence)
                signal = self.generate_signals(divergence)
                self.signal_trading(signal, latest_price)
                time.sleep(60)
            else:
                print("Failed to retrieve latest price. Retrying in 5 seconds...")
                time.sleep(5)


api_key = "1VUueCUnGjzMYK4FGNi7wfWKr19I2sjOrcL31nVyNWOSdvYL6WPhVND7CfHWlOSQVEgJ7Ay648nysS04DbsnHQ"
secret_key = "h8B4G6gVvuz03xNxt9JfxrlQqUbjKX0OFsGsKSms1J1Tw8awuU6aNEYSGHaYUgZpEDG4XGtluOxyVJbyV0UZA"
bot = TradingBot(api_key, secret_key)
bot.run_strategy(bot.run_strategy(symbol="BTCUSDT", start_date="2021-01-01", end_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), timeframe="1h"))

