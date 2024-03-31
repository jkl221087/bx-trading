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

# 設定api鑰駛
api_key = "1VUueCUnGjzMYK4FGNi7wfWKr19I2sjOrcL31nVyNWOSdvYL6WPhVND7CfHWlOSQVEgJ7Ay648nysS04DbsnHQ"
secret_key = "h8B4G6gVvuz03xNxt9JfxrlQqUbjKX0OFsGsKSms1J1Tw8awuU6aNEYSGHaYUgZpEDG4XGtluOxyVJbyV0UZA"

# 創建變量儲存api
market = Market(api_key=api_key, secret_key=secret_key)
trade = Trade(api_key=api_key, secret_key=secret_key)
trade_instance = Trade(api_key=api_key, secret_key=secret_key)

latest_prices_data = None 


#拿到歷史價格
def import_data(symbol, start_str, timeframe):
    client = Client()
    start_str = str(start_str)
    end_str = f"{datetime.now()}"
    df = pd.DataFrame(
        client.get_historical_klines(
            symbol=symbol, interval=timeframe, start_str=start_str, end_str=end_str
        )
    ).astype(float)
    df = df.iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    return df


df = import_data("BTCUSDT", "2021-01-01", "1d")
df = df.iloc[-1000:]

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        )
    ]
)

data = import_data ("BTCUSDT", "2021-01-01", "1h")

data_wick = import_data ("BTCUSDT", "2024-03-01", "1h")



#拿到最新價格
def get_latest_price():
    latest_price_data = market.get_latest_price_of_trading_pair(symbol="BTC-USDT")
    latest_price = float(latest_price_data['price']) if 'price' in latest_price_data else None
    return latest_price


#計算均線
def  calculate_ema (data):
    """
    计算指定窗口大小的指数移动平均线（EMA）
    
    参数：
    - data: 包含收盘价的数据集
    - window: 窗口大小
    
    返回值：
    - ema_50: 50日均线
    - ema_20: 20日均线
    - ema_65: 65日均线
    """

    #計算5日均線
    ema_5 = data['close'].ewm(span = 5, adjust=False).mean()
    # 計算50日均線
    ema_50 = data['close'].ewm(span = 50, adjust=False).mean()
    #計算20日均線
    ema_20 = data['close'].ewm(span = 20, adjust=False).mean()
    #計算65日均線
    ema_60 = data['close'].ewm(span = 65, adjust=False).mean()
    return ema_5, ema_50, ema_20, ema_60

"""計算短期多頭 K線收盤價高於20日與65日的移動平均線之上，且20日平均線 > 65日平均線"""

#計算多頭信號 true or false 
def short_serm_bullish(data):
    """
        計算短期多頭訊號
        
        參數：
        - data: 包含收盤價的資料集
        
        傳回值：
        - bullish_signal: True（短期多頭訊號觸發）或 False（短期多頭訊號未觸發）
        """
    close_prices = data['close']
    
    ma_5 = close_prices.rolling(window = 5).mean()
    ma_10 = close_prices.rolling(window = 10).mean()

    """短期多頭：K線收盤價高於5日與10日的移動平均線之上，且5日平均線 ＞ 10日平均線"""
    if close_prices.iloc[-1] > ma_5.iloc[-1] and close_prices.iloc[-1] > ma_10.iloc[-1] and ma_5.iloc[-1] > ma_10.iloc[-1]:
        bullish_signal = True
    else:
        bullish_signal = False

    return bullish_signal


#判斷支撐壓力
def support_resistance_levels(data, window_size = 10, threshold = 3):
    """判斷支撐壓力"""

    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']

    # 初始化支撑和壓力list
    support_levels = []
    resistance_levels = []

    """體來說，當計算支撐或阻力等級時，window_size 可能表示你希望考慮的過去時間段的長度。
    這個長度可以是以天、小時、分鐘等為單位的時間跨度，根據你的需求和分析的目的而定。
    例如，如果 window_size 設為 30，表示你考慮過去 30
    個數據點的時間範圍來計算支撐或阻力等級。 """
    #遍歷價格數據
    for i in range (len(data)):
        min_price = low_prices [max(0, i - window_size + 1): i + 1].min() #最低價格
        max_price = high_prices [max(0, i - window_size + 1): i + 1].max() #最高價格

        # 計算最大值最小值出現的次數
        mum_min_touches = sum(low_prices [max(0, i - window_size + 1): i + 1] == min_price)
        num_max_touches = sum(high_prices[max(0, i - window_size + 1):i+1] == max_price)

        # 最小值觸及次數超過閾值 加入到support_levels 列表中
        if mum_min_touches >= threshold and mum_min_touches not in support_levels:
            support_levels.append(min_price)

        # 最大值觸及次數超過閾值 加入到resistance_levels 列表中
        if num_max_touches >= threshold and max_price not in resistance_levels:
                resistance_levels.append(max_price)

    return support_levels, resistance_levels


def find_lowest_wick_price(data_wick):
    """
    找出历史中影线长度最低的插针的价格
    
    参数：
    - data_wick: 包含历史价格数据的DataFrame，可能包含高、低、开、收价等信息
    
    返回值：
    - lowest_wick_price: 影线长度最低的插针的价格
    """
    # 计算每个蜡烛的上下影线长度
    data_wick['Upper Wick'] = data_wick['high'] - data_wick[['open', 'close']].max(axis=1)
    data_wick['Lower Wick'] = data_wick[['open', 'close']].min(axis=1) - data_wick['low']

    # 计算插针的长度
    data_wick['Wick Length'] = data_wick[['Upper Wick', 'Lower Wick']].max(axis=1)

    # 找到影线长度最低的插针的索引
    lowest_wick_index = data_wick['Wick Length'].idxmin()

    # 使用索引取出影线长度最低的插针的价格数据
    lowest_wick_price = data_wick.loc[lowest_wick_index]['low']

    return lowest_wick_price



def wick_reversal_strategy(data_wick):
    """根据最新的插针数据来生成买卖信号"""
    # 找出历史影线长度在蜡烛高度的1%到2%之间的最低点插针的价格
    lowest_wick_price = find_lowest_wick_price(data_wick)

    # 获取历史插针的最大长度
    max_historical_wick_length = data_wick['Wick Length'].max()

    # 获取最新蜡烛的高低价
    latest_price = get_latest_price()
    if latest_price is not None:
        # 这里假设最新蜡烛的高价和低价分别是最新价格的 1% 和 2%
        candle_high = latest_price * 1.01
        candle_low = latest_price * 0.98
        
        # 计算最新蜡烛的影线长度
        upper_wick = candle_high - max(latest_price, candle_high)
        lower_wick = min(latest_price, candle_low) - candle_low
        latest_wick_length = max(upper_wick, lower_wick)

        # 计算最大插针长度的1%到2%
        threshold_percentage = 0.01  # 可以调整为1%或2%
        threshold_length = max_historical_wick_length * threshold_percentage

        # 判断最新插针长度是否超过历史最大插针长度的1%到2%
        if latest_wick_length > threshold_length:
            signal = 'buy'  # 如果最新插针长度超过历史最大插针长度的1%到2%，则产生买入信号
        else:
            signal = None   # 否则不产生交易信号
    else:
        print("Failed to retrieve latest price data.")
        signal = None

    return signal







# 交易信号函数不需要传递最新价格，直接返回交易信号即可
def signal_trading(signal, latest_price):
    if signal == 'buy':
        trading_response = get_trading("BTC-USDT", "BUY", "LONG", "MARKET", latest_price)
        print("Trading response:", trading_response)
    elif signal == 'sell':
        trading_response = get_trading("BTC-USDT", "SELL", "SHORT", "MARKET", latest_price)
        print("Trading response:", trading_response)
    else:
        print("No trading signal generated.")

# 初始化
running = True

# 循环判断市场跟下单
while running:
    try:
        latest_price = get_latest_price()
        print("最新價格:", latest_price)
        
        # 执行所有的判断函数
        lowest_wick_price = find_lowest_wick_price(data_wick)
        print("找歷史的最低的插針價格:", lowest_wick_price)
        short = short_serm_bullish(data=data)
        print("短期多頭:", short)
        Support_levels, Resistance_levels = support_resistance_levels(data, window_size=10, threshold=3)
        print("支撐價格:", Support_levels)
        print("壓力價格:", Resistance_levels)
        signal = wick_reversal_strategy(data_wick)
        print("交易訊號:", signal)
        
        # 根据交易信号进行交易
        if signal is not None:
            print("Received signal:", signal)
            signal_trading(signal, latest_price)  # 传递交易信号和最新价格
        else:
            print("No signal received.")
        
        # 延迟时间
        time.sleep(10)  # 15分钟
    except Exception as e:
        print("An error occurred:", str(e))
        # 处理异常，例如重新连接API、记录日志等
        running = False
