import pandas as pd
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗模块
    """
    df_clean = df.copy()
    initial_count = len(df_clean)

    # 时间逻辑清洗
    # 理由：起止时间是后续提取“小时/星期/高峰”特征的唯一依据。行程时间为空，或者下车时间早于上车时间的记录，
    # 通常来自设备故障或系统测试，不属于真实出行数据。
    # 时长<=0无物理意义，>24小时极可能是司机未打表或设备未复位，对城市短时出行需求分析无价值，直接剔除。
    valid_time = (
            df_clean['tpep_pickup_datetime'].notna() &
            df_clean['tpep_dropoff_datetime'].notna() &
            (df_clean['tpep_dropoff_datetime'] > df_clean['tpep_pickup_datetime'])
    )
    duration_sec = (df_clean['tpep_dropoff_datetime'] - df_clean['tpep_pickup_datetime']).dt.total_seconds()
    valid_duration = (duration_sec > 0) & (duration_sec <= 86400)
    df_clean = df_clean[valid_time & valid_duration]

    # 行程距离清洗
    # 理由：距离为0多为乘客取消订单或GPS信号不稳定，导致位置数据出现偏差，这部分记录属于无效数据；>100英里远超纽约市出租车常规运营半径，
    # 属于异常记录，保留会严重压缩后续“距离-车费”散点图的坐标轴，掩盖主体分布规律。
    df_clean = df_clean[(df_clean['trip_distance'] > 0) & (df_clean['trip_distance'] <= 100)]

    # 车费金额清洗
    # 理由：车费<=0通常为系统退款、纠纷单或免单测试记录，不符合正常商业交易逻辑；
    # >500美元为异常记录，会误导模型的学习过程，使模型偏向异常数据，降低整体预测准确性。
    df_clean = df_clean[(df_clean['fare_amount'] > 0) & (df_clean['fare_amount'] <= 500)]

    # 乘客数量清洗
    # 理由：纽约黄色出租车法定载客上限为4-6人，>6违反交通法规且数据字典未定义；
    # <=0不符合物理事实，属录入错误。保留1-6可确保后续“乘客数-车费”箱线图具备真实统计意义。
    df_clean = df_clean[(df_clean['passenger_count'] >= 1) & (df_clean['passenger_count'] <= 6)]

    # 区域ID清洗
    # 理由：NYC TLC官方Taxi Zone Lookup Table仅包含1-265个有效编码。超出此范围的ID无法关联到具体行政区/街区，
    # 会引入统计误差，使地理聚合分析和热度图无法准确反映真实分布。
    df_clean = df_clean[
        df_clean['PULocationID'].between(1, 265) &
        df_clean['DOLocationID'].between(1, 265)
        ]

    # 支付方式清洗
    # 理由：官方字典明确定义0-6为合法编码。
    # 超出范围的数值无法进行支付方式分布统计，直接剔除以保证分类特征纯净。
    df_clean = df_clean[df_clean['payment_type'].between(0, 6)]

    # 缺失值兜底处理
    # 理由：经过前面几步的业务过滤，剩余的缺失值已经很少。为了避免时间提取和模型训练时因空值报错，
    # 这里直接删除所有仍含缺失值的行。数据整体质量较高，直接删除比随意填充更稳妥，也不会影响分析结论。
    df_clean = df_clean.dropna()

    # 索引重置与内存回收
    # 理由：多轮过滤后，数据的行号不再连续，会影响后续按位置选取数据和时间段的划分，所以需要重置索引来保持数据连贯。
    df_clean = df_clean.reset_index(drop=True)

    # 打印清洗摘要
    final_count = len(df_clean)
    drop_rate = ((initial_count - final_count) / initial_count) * 100
    print(f"[M1] 数据清洗完成 | 原始: {initial_count:,} 行 → 清洗后: {final_count:,} 行 | 剔除率: {drop_rate:.2f}%")

    return df_clean


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    时间特征提取
    """
    df_feat = df.copy()

    # 提取上车小时
    df_feat['pickup_hour'] = df_feat['tpep_pickup_datetime'].dt.hour.astype('Int8')

    # 提取星期几
    df_feat['pickup_dayofweek'] = df_feat['tpep_pickup_datetime'].dt.dayofweek.astype('Int8')

    # 生成是否周末标签
    df_feat['is_weekend'] = (df_feat['pickup_dayofweek'] >= 5).astype('Int8')

    # 生成是否高峰时段标签
    is_morning_peak = df_feat['pickup_hour'].between(7, 9)
    is_evening_peak = df_feat['pickup_hour'].between(17, 19)
    df_feat['is_peak_hour'] = (is_morning_peak | is_evening_peak).astype('Int8')

    print(f"[M1] 时间特征提取完成 | 新增: pickup_hour, pickup_dayofweek, is_weekend, is_peak_hour")
    return df_feat

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    衍生特征设计
    """
    df_derived = df.copy()

    # 衍生特征1：行程时长
    # 直接反映路况拥堵程度与司机运营效率。相同时段/距离下，时长越长说明拥堵越严重，
    # 越容易产生等候费或动态加价。这个特征一方面能支撑 M2 对高峰时段的分布分析，另一方面在 M3 中可作为重要的时间变量，
    # 帮助模型理解拥堵如何影响需求变化。
    duration_sec = (df_derived['tpep_dropoff_datetime'] - df_derived['tpep_pickup_datetime']).dt.total_seconds()
    df_derived['trip_duration_min'] = (duration_sec / 60.0).astype('float32')

    # 衍生特征2：单位距离车费
    # 剔除距离影响后得到的平均费用水平，正常范围约在每英里2.5美元到8美元之间。如果明显偏高，
    # 可能反映拥堵导致的计时费增加、机场或夜间附加费、绕路或溢价时段等原因。该特征可作为M3车费影响因素分析的辅助变量，
    # 也可在M4问答中作为估算费用的参考基准。为防止极值干扰，对距离做了最小值保护并截断极端值。
    safe_dist = df_derived['trip_distance'].clip(lower=0.01)
    df_derived['fare_per_mile'] = (df_derived['fare_amount'] / safe_dist).astype('float32')
    q99 = df_derived['fare_per_mile'].quantile(0.99)
    df_derived['fare_per_mile'] = df_derived['fare_per_mile'].clip(upper=q99)

    print(f"[M1]  衍生特征构建完成 | 新增: trip_duration_min, fare_per_mile (99分位截断: ${q99:.2f}/mile)")
    return df_derived


def save_cleaned_data(df: pd.DataFrame, data_dir: str = "data") -> str:
    """保存清洗与特征工程后的数据"""
    base_dir = Path(__file__).resolve().parent.parent
    save_path = base_dir / data_dir / "cleaned_taxi_data.parquet"
    # 避免写入无用行索引列
    df.to_parquet(save_path, index=False, engine='pyarrow')
    print(f"[M1] 清洗后数据已保存至: {save_path} | 大小: {save_path.stat().st_size / (1024**2):.2f} MB")
    return str(save_path)


def run_m1() -> pd.DataFrame:
    """加载 → 报告 → 清洗 → 时间特征 → 衍生特征 → 保存"""
    print("\n" + "="*50)
    print("启动 M1 数据处理模块")
    print("="*50)
    df_raw = load_data()
    generate_quality_report(df_raw)
    df_cleaned = clean_data(df_raw)
    df_time = extract_time_features(df_cleaned)
    df_final = create_derived_features(df_time)
    save_cleaned_data(df_final)
    print("M1 模块执行完毕，返回完整特征的数据。\n")
    return df_final
