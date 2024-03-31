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

# 设置 API 密钥
api_key = "1VUueCUnGjzMYK4FGNi7wfWKr19I2sjOrcL31nVyNWOSdvYL6WPhVND7CfHWlOSQVEgJ7Ay648nysS04DbsnHQ"
secret_key = "h8B4G6gVvuz03xNxt9JfxrlQqUbjKX0OFsGsKSms1J1Tw8awuU6aNEYSGHaYUgZpEDG4XGtluOxyVJbyV0UZA"

# 创建 Market 和 Trade 对象时直接使用变量中存储的 API 密钥
market = Market(api_key=api_key, secret_key=secret_key)
trade = Trade(api_key=api_key, secret_key=secret_key)
trade_instance = Trade(api_key=api_key, secret_key=secret_key)

latest_prices_data = None 


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




def get_latest_prices():
    global latest_prices_data
    latest_prices_data = market.get_latest_price_of_trading_pair(symbol="BTC-USDT")
    return latest_prices_data


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

def short_serm_bullish(data):
    """
    计算短期多头信号
    
    参数：
    - data: 包含收盘价的数据集
    
    返回值：
    - bullish_signal: True（短期多头信号触发）或 False（短期多头信号未触发）
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
def support_resistance_levels(data):
    """判斷支撐壓力"""

    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']

    #計算支撐水平(取的最低價)
    support_level = low_prices.min()
    
    #計算壓力水平(取的最高的價格)
    resistance_level = high_prices.max()
    return support_level, resistance_level

#找到市场快速下跌时插针最深的位置
def find_deepest_wick(data):
    """
    找到市场快速下跌时插针最深的位置
    
    参数：
    - data: 包含高、低、开、收价的数据集
    
    返回值：
    - deepest_wick_index: 插针最深的位置的索引
    """
    #計算每個蠟燭的上下影線長度
    data['Upper Wick'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['Lower Wick'] = data[['open', 'close']].min(axis=1) - data['low']

    # 計算插針的深度
    data['Wick Length'] = data[['Upper Wick', 'Lower Wick']].max(axis=1)

    #找到插針的位置

    deepest_wick_index = data['Wick Length'].idxmax()

    print(deepest_wick_index)


# 根据市场插针产生交易信号
def wick_signal(latest_prices, deepest_wick_index):
    """
    根据市场快速到达插针位置的情况生成交易信号
    
    参数：
    - latest_prices: 最新价格数据，可能包含开、高、低、收价以及交易量等信息
    - deepest_wick_index: 插针最深的位置的索引
    
    返回值：
    - signal: 交易信号 ('buy' 或 'sell')
    """
    # 检查deepest_wick_index是否在latest_prices数据的有效索引范围内
    deepest_wick_index = int(find_deepest_wick(data))
    if deepest_wick_index >= 0 and deepest_wick_index < len(latest_prices):
        # 取的插针的位置
        wick_data = latest_prices[deepest_wick_index]

        # 如果市场快速到达插针的位置 
        if wick_data['Volume'] > some_threshold:  # 市场快速到达插针位置的条件
            # 如果市场是快速下跌到插针的位置
            if wick_data['close'] < wick_data['open']:
                # 检查下一个蜡烛的开盘价是否高于当前蜡烛的最高价
                next_candle_index = deepest_wick_index + 1
                if next_candle_index < len(latest_prices):  # 确保下一根蜡烛的索引在有效范围内
                    next_candle = latest_prices[next_candle_index]
                    if next_candle['open'] > wick_data['high']:
                        return 'buy'  # 在下一根蜡烛进场做多
            # 如果市场是快速上涨到插针的位置
            elif wick_data['close'] > wick_data['open']:
                # 检查下一个蜡烛的开盘价是否低于当前蜡烛的最低价
                next_candle_index = deepest_wick_index + 1
                if next_candle_index < len(latest_prices):  # 确保下一根蜡烛的索引在有效范围内
                    next_candle = latest_prices[next_candle_index]
                    if next_candle['open'] < wick_data['low']:
                        return 'sell'  # 在下一根蜡烛进场做空
    return None  # 没有交易信号，返回None




def signal_trading(latest_prices, deepest_wick_index):
    # 计算其他指标、生成交易信号等操作...
    
    # 根据交易信号执行交易
    trade_instance = Trade(api_key=api_key, secret_key=secret_key)
    signal = wick_signal(latest_prices, deepest_wick_index)
    if signal == 'buy':
        trading_response = get_trading("BTC-USDT", "BUY", "LONG", "MARKET", 10000,)
        print("Trading response:", trading_response)
    elif signal == 'sell':
        trading_response = get_trading("ID-USDT", "SELL", "SHORT", "MARKET", 10000)
        print("Trading response:", trading_response)
    else:
        print("No trading signal generated.")

    return latest_prices, deepest_wick_index



# 假设 some_threshold 是一个阈值，用于判断市场快速到达插针位置的条件
some_threshold = 10000

running = True



while running:
    open_orders = trade_instance.get_open_orders(symbol="BTC-USDT")
    print("Open Orders:", open_orders)
    """if open_orders['orders']"""

    bullish_signal = short_serm_bullish(data)
    print("短期多头信号:", bullish_signal)

    # 获取最新价格数据和插针位置
    latest_prices = get_latest_prices()
    deepest_wick_index = find_deepest_wick(data)
    print("Latest prices:", latest_prices)
    print("Wick index:", deepest_wick_index)

    # 执行信号交易
    latest_prices, wick_index = signal_trading(latest_prices, deepest_wick_index)
    """在这里添加条件来决定是否继续循环执行
    比如设定一个循环次数，或者根据某些市场条件来决定是否继续执行循环
    当满足退出循环条件时，将 running 设置为 False
    例如：
    if 满足退出循环条件:
        running = False"""
    time.sleep(10) 

support_resistance_levels(data = data)
calculate_ema(data = data)
short_serm_bullish(data = data)
find_deepest_wick(data = data)
fvg_data = smc.fvg(df)
swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=50)
bos_choch_data = smc.bos_choch(df, swing_highs_lows_data)
ob_data = smc.ob(df, swing_highs_lows_data)
liquidity_data = smc.liquidity(df, swing_highs_lows_data)
previous_high_low_data = smc.previous_high_low(df, time_frame="1W")
# fig = add_FVG(fig, fvg_data)
# fig = add_swing_highs_lows(fig, swing_highs_lows_data)
# fig = add_bos_choch(fig, bos_choch_data)
# fig = add_OB(fig, ob_data)
# fig = add_liquidity(fig, liquidity_data)
# fig = add_previous_high_low(fig, previous_high_low_data)

# fig.update_layout(xaxis_rangeslider_visible=False)
# fig.update_layout(showlegend=False)
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# fig.update_xaxes(visible=False, showticklabels=False)
# fig.update_yaxes(visible=False, showticklabels=False)
# fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
# fig.update_layout(paper_bgcolor="rgba(94, 94, 134, 1)")
# fig.update_layout(font=dict(color="white"))
# fig.write_image("test_binance.png")