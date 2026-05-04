import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def build_prediction_dataset(df: pd.DataFrame) -> tuple:
    """
    构建需求预测数据集 + 时序划分与标准化
    """
    print("[M3] 正在构建预测数据集...")

    # 聚合需求标签
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    demand_df = df.groupby(['PULocationID', 'pickup_date', 'pickup_hour']).size().reset_index(name='demand')

    # 按区域与时间排序
    demand_df = demand_df.sort_values(['PULocationID', 'pickup_date', 'pickup_hour']).reset_index(drop=True)

    # 构造基础时空特征
    demand_df['date'] = pd.to_datetime(demand_df['pickup_date'])
    demand_df['dayofweek'] = demand_df['date'].dt.dayofweek.astype('Int8')
    demand_df['is_weekend'] = (demand_df['dayofweek'] >= 5).astype('Int8')
    demand_df['is_peak_hour'] = (
                demand_df['pickup_hour'].between(7, 9) | demand_df['pickup_hour'].between(17, 19)).astype('Int8')

    # 构造滞后特征
    demand_df['lag_1h'] = demand_df.groupby('PULocationID')['demand'].shift(1)
    demand_df['lag_24h'] = demand_df.groupby('PULocationID')['demand'].shift(24)

    # 剔除滞后产生的 NaN 行
    demand_df = demand_df.dropna().reset_index(drop=True)

    # 准备特征矩阵 X 与标签 y
    feature_cols = ['PULocationID', 'pickup_hour', 'dayofweek', 'is_weekend', 'is_peak_hour', 'lag_1h', 'lag_24h']
    X = demand_df[feature_cols].copy()
    y = demand_df['demand'].values

    # 全局按时间排序并划分训练/测试集 (8:2)
    demand_df = demand_df.sort_values(['date', 'pickup_hour']).reset_index(drop=True)
    split_idx = int(len(demand_df) * 0.8)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 数值特征标准化
    scaler = StandardScaler()
    num_cols = ['pickup_hour', 'dayofweek', 'lag_1h', 'lag_24h']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 转为 float32 适配 PyTorch/TF 张量计算
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.astype(np.float32)
    y_test_np = y_test.astype(np.float32)

    print(f"[M3] 数据集构建完成 | 训练集: {len(X_train_np):,} 样本 | 测试集: {len(X_test_np):,} 样本")
    print(
        f"[M3] 特征维度: {X_train_np.shape[1]} | 标签分布: 均值={y_train_np.mean():.1f}, 标准差={y_train_np.std():.1f}")

    return X_train_np, X_test_np, y_train_np, y_test_np, scaler, feature_cols


def run_m3(df: pd.DataFrame) -> dict:
    """
    模块统一调度入口
    """
    print("\n" + "=" * 50)
    print("启动 M3 预测模型模块")
    print("=" * 50)
    X_train, X_test, y_train, y_test, scaler, features = build_prediction_dataset(df)
    print("M3 阶段1&2执行完毕，数据集已就绪。\n")
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'features': features
    }
