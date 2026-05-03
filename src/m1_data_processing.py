import pandas as pd
import os
from pathlib import Path


def load_data(data_dir: str = "data", filename: str = "yellow_tripdata_2023-01.parquet") -> pd.DataFrame:
    """
    加载纽约黄色出租车 PARQUET 数据
    策略：按需读取列 + 类型降级 + pyarrow引擎，防止300万行数据导致内存溢出
    """
    # 路径解析
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / data_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {file_path}\n"
            f"请从 NYC TLC 官网下载 2023年1月 Yellow Taxi 数据，\n"
            f"   重命名为 '{filename}' 并放入项目根目录的 data/ 文件夹中。"
        )

    # 定义核心字段
    # tpep_pickup_datetime / tpep_dropoff_datetime : 起止时间
    # PULocationID / DOLocationID                  : 上下客区域ID
    # passenger_count / trip_distance              : 乘客数 / 行程距离(英里)
    # fare_amount / tip_amount / total_amount      : 车费 / 小费 / 总金额
    # payment_type                                 : 支付方式 (1=信用卡,2=现金等)
    use_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'PULocationID', 'DOLocationID',
        'passenger_count', 'trip_distance',
        'fare_amount', 'tip_amount', 'total_amount',
        'payment_type'
    ]

    print(f"[M1] 正在加载数据: {file_path.name} ...")

    # 读取 PARQUET
    df = pd.read_parquet(file_path, columns=use_cols, engine='pyarrow')

    # 数据类型降级防内存溢出
    dtype_config = {
        'passenger_count': 'Int8',
        'payment_type': 'Int8',
        'PULocationID': 'Int16',
        'DOLocationID': 'Int16',
        'trip_distance': 'float32',
        'fare_amount': 'float32',
        'tip_amount': 'float32',
        'total_amount': 'float32'
    }
    df = df.astype(dtype_config)

    # 时间戳标准化
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # 打印加载摘要
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[M1] 数据加载完成 | 形状: {df.shape[0]:,} 行 × {df.shape[1]} 列 | 内存占用: {mem_mb:.2f} MB")

    return df
