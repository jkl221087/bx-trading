import csv
from market import Market
from datetime import datetime, timedelta

market = Market(api_key="1VUueCUnGjzMYK4FGNi7wfWKr19I2sjOrcL31nVyNWOSdvYL6WPhVND7CfHWlOSQVEgJ7Ay648nysS04DbsnHQ", 
                secret_key="h8B4G6gVvuz03xNxt9JfxrlQqUbjKX0OFsGsKSms1J1Tw8awuU6aNEYSGHaYUgZpEDG4XGtluOxyVJbyV0UZA")

def kline(start_date=None, end_date=None, output_file="kline_data.csv"):
    # 将日期字符串转换为时间戳
    def date_to_timestamp(date_str):
        if date_str:
            date_object = datetime.strptime(date_str, "%Y-%m-%d")
            return int(date_object.timestamp()) * 1000
        else:
            return None

    # 将日期字符串转换为时间戳
    start_time = date_to_timestamp(start_date)
    end_time = date_to_timestamp(end_date)

    # 打开 CSV 文件进行写入
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入 CSV 文件的标题行
        writer.writerow(["Time", "Open", "High", "Low", "Close", "Volume","Signal"])

        # 获取价格数据的函数
        def get_prices(start_time, end_time):
            current_time = start_time
            while current_time < end_time:
                # 计算下一个时间间隔的结束时间
                next_time = current_time + 3600000  # 1小时的毫秒数

                # 调用 market.get_k_line_data 方法并传递 current_time 和 next_time 参数
                k_line_data = market.get_k_line_data(symbol="BTC-USDT", interval="1h", start_time=current_time, end_time=next_time)

                # 将价格数据写入 CSV 文件
                for row in k_line_data:
                    # 将时间戳转换为日期时间字符串，使用指定的格式
                    time_str = datetime.utcfromtimestamp(row["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([time_str, row["open"], row["high"], row["low"], row["close"], row["volume"]])

                # 更新 current_time 为下一个时间间隔的开始时间
                current_time = next_time

        # 调用获取价格数据的函数
        get_prices(start_time, end_time)

# 指定开始日期和结束日期
start_date = "2024-01-01"
end_date = "2024-03-31"

# 调用 kline 函数并传递开始日期、结束日期和输出文件名参数
kline(start_date=start_date, end_date=end_date, output_file="kline_data.csv")
