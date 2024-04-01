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

# 找出插針價格
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

#找出最新的插針價格
def find_latest_wick_length(data):
    """计算最新插针的长度"""
    # 获取最新一根蜡烛的高价和低价
    latest_high = data.iloc[-1]['high']
    latest_low = data.iloc[-1]['low']

    # 计算最新蜡烛的上下影线长度
    upper_wick = latest_high - max(latest_high, latest_low)
    lower_wick = min(latest_high, latest_low) - latest_low

    # 计算最新插针的长度
    latest_wick_length = max(upper_wick, lower_wick)

    return latest_wick_length


def calculate_slope(data, window):
    """
    计算移动平均线的斜率
    
    参数：
    - data: 包含价格数据的数组或列表
    - window: 移动平均线的窗口大小
    
    返回值：
    - 斜率值，正数表示上升趋势，负数表示下降趋势，接近于零表示横盘
    """
    # 计算移动平均线
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 计算斜率
    slope = np.gradient(ma)
    
    # 返回斜率的最后一个值，即最新的斜率值
    return slope[-1]

#計算背離
def calculate_divergence(prices, indicator):
    """
    计算价格与指标之间的背离
    
    参数：
    - prices: 包含价格数据的列表或数组
    - indicator: 包含指标数据的列表或数组，长度与prices相同
    
    返回值：
    - divergence: 背离的数量，如果出现背离则为正值，否则为负值
    """
    if len(prices) != len(indicator):
        raise ValueError("Prices and indicator lengths must be equal.")
    
    # 计算价格和指标的变化
    price_changes = np.diff(prices)
    indicator_changes = np.diff(indicator)
    
    # 计算价格和指标的符号
    price_signs = np.sign(price_changes)
    indicator_signs = np.sign(indicator_changes)
    
    # 计算价格和指标变化的乘积
    product = price_signs * indicator_signs
    
    # 统计背离的数量
    divergence = np.sum(product < 0)
    
    return divergence

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

def macd_signal(data, short_window, long_window, signal_window):
    """
    计算MACD指标的金叉死叉信号
    
    参数：
    - data: 包含历史价格数据的DataFrame，至少包含'Close'列
    - short_window: 短期EMA的窗口大小
    - long_window: 长期EMA的窗口大小
    - signal_window: Signal线的窗口大小
    
    返回值：
    - macd_signal: 包含MACD指标金叉死叉信号的Series
    """
    # 计算短期和长期EMA
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    # 计算MACD线
    macd_line = short_ema - long_ema
    
    # 计算Signal线
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # 计算MACD的差值
    macd_diff = macd_line - signal_line
    
    # 计算金叉死叉信号
    macd_signal = np.where(macd_diff > 0, 1, -1)
    
    return macd_signal


def combined_signal(macd_signal, volume_momentum, volume_data):
    """
    结合MACD指标和成交量动量产生交易信号
    
    参数：
    - macd_signal: 包含MACD指标金叉死叉信号的Series
    - volume_momentum: 包含成交量动量的Series
    - volume_data: 包含历史成交量数据的DataFrame
    
    返回值：
    - trade_signal: 交易信号，可能是'buy'、'sell'或'hold'
    """
    # 获取最新成交量和平均成交量
    latest_volume = volume_data.iloc[-1]['Volume']
    avg_volume = data['Volume'].mean()
    
    # 如果MACD出现金叉并且成交量动量为正且成交量高于平均值，则发出买入信号
    if macd_signal[-1] == 1 and volume_momentum[-1] > 0 and latest_volume > avg_volume:
        trade_signal = 'buy'
    # 如果MACD出现死叉并且成交量动量为负且成交量高于平均值，则发出卖出信号
    elif macd_signal[-1] == -1 and volume_momentum[-1] < 0 and latest_volume > avg_volume:
        trade_signal = 'sell'
    # 其他情况下保持持有头寸
    else:
        trade_signal = 'hold'
    
    return trade_signal


# MACD signal
def generate_signal(macd_line, signal_line, momentum):
    """
    根据MACD线、信号线和动量的情况生成交易信号

    参数：
    - macd_line: MACD线的数值列表或数组
    - signal_line: 信号线的数值列表或数组
    - momentum: 动量的数值，例如市场动量或其他指标

    返回值：
    - signal: 交易信号，可能的取值为'BUY'（买入）、'SELL'（卖出）或'NONE'（无信号）
    """
    if momentum > 0:  # 如果市场处于上升动量状态
        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
            signal = 'BUY'  # MACD线上穿信号线，产生买入信号
        else:
            signal = 'NONE'  # 其他情况无信号
    elif momentum < 0:  # 如果市场处于下降动量状态
        if macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
            signal = 'SELL'  # MACD线下穿信号线，产生卖出信号
        else:
            signal = 'NONE'  # 其他情况无信号
    else:  # 如果市场处于横盘或静止状态
        signal = 'NONE'  # 无信号

    return signal

# 背離signal
def generate_signals(prices, indicator):
    """
    根据价格和指标的背离生成买入或卖出信号
    
    参数：
    - prices: 包含价格数据的列表或数组
    - indicator: 包含指标数据的列表或数组，长度与prices相同
    
    返回值：
    - signals: 包含买入或卖出信号的列表，与prices和indicator的长度相同
    """
    divergence = calculate_divergence(prices, indicator)
    signals = []
    for div in divergence:
        if div > 0:
            signals.append('Buy')
        elif div < 0:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

def signal(prices, indicator, macd_line, signal_line, momentum, volume_data):
    """
    根据价格、指标、MACD线、信号线和动量生成交易信号
    
    参数：
    - prices: 包含价格数据的列表或数组
    - indicator: 包含指标数据的列表或数组，长度与prices相同
    - macd_line: MACD线的数值列表或数组
    - signal_line: 信号线的数值列表或数组
    - momentum: 动量的数值，例如市场动量或其他指标
    - volume_data: 包含历史成交量数据的DataFrame
    
    返回值：
    - trade_signal: 交易信号，可能是'buy'、'sell'或'hold'
    """
    # 生成MACD交易信号
    macd_trade_signal = generate_signal(macd_line, signal_line, momentum)
    
    # 生成背离信号
    divergence_signals = generate_signals(prices, indicator)
    
    # 结合MACD交易信号和背离信号
    combined_signal = combined_signal(macd_trade_signal, momentum, volume_data)
    
    return combined_signal




def signal_trading(signal, latest_price):
    """
    根据交易信号执行交易
    
    参数：
    - signal: 交易信号，可能是'buy'、'sell'或'hold'
    - latest_price: 最新价格，用于市价单
    
    返回值：
    - 无
    """
    if signal == 'buy':
        trading_response = get_trading("BTC-USDT", "BUY", "LONG", "MARKET", latest_price)
        print("Trading response:", trading_response)
    elif signal == 'sell':
        trading_response = get_trading("BTC-USDT", "SELL", "SHORT", "MARKET", latest_price)
        print("Trading response:", trading_response)
    else:
        print("No trading signal generated.")



capital = 10000
position = 0
trade_history = [] 

# 初始化
running = True

# 假设你有一个包含收盘价的价格数据列表 price_data
price_data = data  # 你的价格数据

# 设置移动平均线的窗口大小
window_size = 10  # 这里假设窗口大小为 10

# 调用 calculate_slope 函数计算斜率
slope = calculate_slope(price_data, window_size)

# 动量为斜率值，即移动平均线的斜率
momentum = slope

closes = data['close'].tolist()
macd_line = calculate_macd(closes)
print("MACD：", macd_line)
signal_line = calculate_macd_signal_line(closes)
print("MACD的信号线：", signal_line)
Generate_signal = generate_signal(macd_line, signal_line, momentum)


# # 循环判断市场跟下单
# while running:
#     try:
#         latest_price = get_latest_price()
#         print("最新價格:", latest_price)
        
#         # 执行所有的判断函数
#         lowest_wick_price = find_lowest_wick_price(data_wick)
#         print("找歷史的最低的插針價格:", lowest_wick_price)
#         short = short_serm_bullish(data=data)
#         print("短期多頭:", short)
#         Support_levels, Resistance_levels = support_resistance_levels(data, window_size=10, threshold=3)
#         print("支撐價格:", Support_levels)
#         print("壓力價格:", Resistance_levels)
#         Signal = signal()
#         print("交易訊號:", signal)
        
#         # 根据交易信号进行交易
#         if signal is not None:
#             print("Received signal:", signal)
#             signal_trading(signal, latest_price)  # 传递交易信号和最新价格
#         else:
#             print("No signal received.")

#         # 计算策略收益率
#         initial_capital = 10000
#         final_capital = capital + position * latest_price  # 计算最后持有头寸的价值
#         profit = final_capital - initial_capital
#         return_percentage = profit / initial_capital * 100
#         print("Strategy return percentage:", return_percentage)

#         # 绘制交易历史
#         buys = [trade[1] for trade in trade_history if trade[0] == 'buy']
#         sells = [trade[1] for trade in trade_history if trade[0] == 'sell']
#         buy_times = [trade[2] for trade in trade_history if trade[0] == 'buy']
#         sell_times = [trade[2] for trade in trade_history if trade[0] == 'sell']


#         plt.plot(buy_times, buys, 'go', label='Buy')
#         plt.plot(sell_times, sells, 'ro', label='Sell')
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title('Trading History')
#         plt.legend()
#         plt.show()
        
#         # 延迟时间
#         time.sleep(10)  # 15分钟
#     except Exception as e:
#         print("An error occurred:", str(e))
#         # 处理异常，例如重新连接API、记录日志等
#         running = False






