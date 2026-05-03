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


def generate_quality_report(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """
    生成数据质量报告
    """
    base_dir = Path(__file__).resolve().parent.parent
    out_path = base_dir / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "data_quality_report.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("纽约黄色出租车数据质量报告 (2023年1月)")
    lines.append("=" * 60)
    lines.append(f"总记录数: {len(df):,} 条\n")

    # 缺失率统计
    lines.append("1. 缺失率统计 (Missing Rate)")
    lines.append("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in df.columns:
        lines.append(f"{col:<25} | 缺失: {missing[col]:>6} ({missing_pct[col]:5.2f}%)")
    lines.append("")

    # 异常值统计
    lines.append("2. 异常值统计 (Outlier Detection)")
    lines.append("-" * 40)
    total = len(df)
    outliers = {}

    # 时间逻辑异常
    invalid_time = df['tpep_pickup_datetime'].isna() | df['tpep_dropoff_datetime'].isna()
    time_reverse = df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']
    duration_h = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    duration_abnormal = ((duration_h <= 0) | (duration_h > 24)).fillna(False)
    outliers['时间异常(NaT/倒挂/时长<0或>24h)'] = (invalid_time | time_reverse | duration_abnormal).sum()

    # 行程距离异常
    outliers['行程距离异常(<=0 或 >100英里)'] = ((df['trip_distance'] <= 0) | (df['trip_distance'] > 100)).sum()

    # 车费异常
    outliers['车费异常(<=0 或 >500美元)'] = ((df['fare_amount'] <= 0) | (df['fare_amount'] > 500)).sum()

    # 乘客数异常
    outliers['乘客数异常(<=0 或 >6人)'] = ((df['passenger_count'] <= 0) | (df['passenger_count'] > 6)).sum()

    # 区域 ID 异常
    outliers['上下客区域ID异常(非1-265)'] = (
            (~df['PULocationID'].between(1, 265)) | (~df['DOLocationID'].between(1, 265))
    ).sum()

    # 支付方式异常
    outliers['支付方式异常(非0-6)'] = (~df['payment_type'].between(0, 6)).sum()

    for rule, count in outliers.items():
        pct = (count / total) * 100
        lines.append(f"{rule:<35} | 异常数: {count:>6} ({pct:5.2f}%)")

    lines.append("")
    lines.append("   说明: 异常值统计存在交叉，实际清洗将采用并集过滤策略。")
    lines.append("   报告已自动保存，供M1.3清洗策略参考。")
    lines.append("=" * 60)

    report_content = "\n".join(lines)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"[M1] 数据质量报告已保存至: {report_file}")
    print(report_content)
    return str(report_file)
