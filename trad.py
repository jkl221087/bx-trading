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

prices = []


for index, row in data_wick.iterrows():
    prices.append(row['close'])

#拿到最新價格
def get_latest_price():
    latest_price_data = market.get_latest_price_of_trading_pair(symbol="BTC-USDT")
    latest_price = float(latest_price_data['price']) if 'price' in latest_price_data else None
    return latest_price


def calculate_ema(prices):
    """
    计算指定窗口大小的指数移动平均线（EMA）

    参数：
    - prices: 包含价格数据的列表或数组

    返回值：
    - ema_5: 5日均线
    - ema_50: 50日均线
    - ema_20: 20日均线
    - ema_65: 65日均线
    """

    # 将价格转换为 Pandas Series 对象
    prices_series = pd.Series(prices)

    # 计算5日均线
    ema_5 = prices_series.ewm(span=5, adjust=False).mean()
    # 计算50日均线
    ema_50 = prices_series.ewm(span=50, adjust=False).mean()
    # 计算20日均线
    ema_20 = prices_series.ewm(span=20, adjust=False).mean()
    # 计算65日均线
    ema_65 = prices_series.ewm(span=65, adjust=False).mean()
    return ema_5, ema_50, ema_20, ema_65

# 计算价格的EMA
ema_5, ema_50, ema_20, ema_65 = calculate_ema(prices)
# print("EMA 5:", ema_5)
# print("EMA 50:", ema_50)
# print("EMA 20:", ema_20)
# print("EMA 65:", ema_65)

indicator = calculate_ema(prices)

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


#計算背離
def calculate_divergence(prices, indicator):
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
        divergence = "顶部背离"
    elif price_change_sign == -1 and indicator_change_sign == 1:
        divergence = "底部背离"
    else:
        divergence = "无背离"

    return divergence

ema_20_list = ema_20.values.tolist()
divergence = calculate_divergence(prices, ema_20_list)
print('背離类型:', divergence)

#計算成交量
def volume_momentum(data, lookback_period):
    """
    计算成交量动量

    参数：
    - data: 包含历史价格数据的DataFrame，至少包含'Volume'列
    - lookback_period: 动量计算的回看期

    返回值：
    - volume_momentum: 包含成交量动量的Series
    """
    # 计算成交量的变化
    volume_change = data['Volume'].diff(lookback_period)

    return volume_change

# 計算 macd line
def calculate_macd(closes):
    # Step 1: 计算12日EMA
    alpha_12 = 2 / (12 + 1)
    ema_12 = closes.ewm(span=12, adjust=False).mean()
    
    # Step 2: 计算26日EMA
    alpha_26 = 2 / (26 + 1)
    ema_26 = closes.ewm(span=26, adjust=False).mean()
    
    # Step 3: 计算DIF（差离值）
    dif = ema_12 - ema_26
    
    # Step 4: 计算9日EMA
    alpha_9 = 2 / (9 + 1)
    ema_9 = dif.ewm(span=9, adjust=False).mean()
    
    # Step 5: 计算MACD线
    macd_line = dif - ema_9
    
    return macd_line

def calculate_macd_signal_line(closes):
    # 计算DIF线
    ema_12 = closes.ewm(span=12, adjust=False).mean()
    ema_26 = closes.ewm(span=26, adjust=False).mean()
    dif = ema_12 - ema_26
    
    # 计算DIF的9日EMA作为信号线
    signal_line = dif.ewm(span=9, adjust=False).mean()
    
    return signal_line

def generate_signals(indicator):
    """
    根据背离类型生成买入或卖出信号

    参数：
    - indicator: 背离类型，字符串 "顶部背离"、"底部背离" 或 "无背离"

    返回值：
    - signal: 买入或卖出信号，"Buy"、"Sell" 或 "Hold"
    """
    if indicator == "顶部背离":
        signal = 'SELL'
    elif indicator == "底部背离":
        signal = 'BUY'
    else:
        signal = 'Hold'
    return signal

# 生成信号
signal = generate_signals(divergence)
print(signal)



def signal_trading(signal, latest_price, position):
    """
    根据交易信号执行交易
    
    参数：
    - signal: 交易信号，可能是'BUY'、'SELL'或'HOLD'
    - latest_price: 最新价格，用于市价单
    - position: 当前仓位，1 表示持有多头仓位，-1 表示持有空头仓位，0 表示没有持仓
    
    返回值：
    - position: 更新后的仓位
    """
# 计算止损价（假设为最新价格的0.5%）
    stop_loss_price = latest_price // 0.98
    print(stop_loss_price)
    # 设置止盈比例

    # 根据最新价格计算止盈价
    take_profit_price = latest_price * 0.99
    print(take_profit_price)
# 0.5% 的止损价

    # 计算止盈价（假设为最新价格的1.005倍）

    
    take_profit_payload = f'{{"type": "TAKE_PROFIT_MARKET", "stopPrice": {take_profit_price}, "price": {take_profit_price}, "workingType": "MARK_PRICE"}}'
    stop_loss_payload = f'{{"type": "STOP_MARKET", "stopPrice": {stop_loss_price}, "price": {stop_loss_price}, "workingType": "MARK_PRICE"}}'

    # 如果已经有仓位，就不再下单
    if position != 0:
        print("Already have position, no trading needed.")
        return position

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

    return position





# 初始化
running = True
position = 0

while running:
    # 获取最新价格
    latest_price = get_latest_price()

    if latest_price is not None:

        latest_data = import_data("BTCUSDT", "2024-03-01", "1h")
        latest_prices = latest_data['close']
        
        ema_5, ema_50, ema_20, ema_65 = calculate_ema(latest_prices)
        ema_20_list = ema_20.values.tolist()
        divergence = calculate_divergence(latest_prices, ema_20_list)
        
        # 生成交易信号
        signal = generate_signals(divergence)
        
        # 执行交易
        signal_trading(signal, latest_price, position)
        
        # 等待一段时间再继续循环
        time.sleep(60)  # 每隔一分钟检测一次市场
    else:
        print("Failed to retrieve latest price. Retrying in 5 seconds...")
        time.sleep(10)  # 如果获取最新价格失败，则等待5秒钟后重试


