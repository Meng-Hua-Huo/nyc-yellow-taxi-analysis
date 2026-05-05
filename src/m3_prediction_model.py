import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


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


class DemandNN(nn.Module):
    """
    需求预测 MLP 神经网络
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_nn_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=512, lr=0.001, patience=5) -> dict:
    """
    训练循环与 Loss 曲线监控
    """
    print("[M3] 正在初始化并训练 PyTorch 神经网络...")

    # 设备与随机种子配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    # 构建 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数与优化器初始化
    model = DemandNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_dataset)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val)
                epoch_val_loss += criterion(val_preds, y_val).item() * X_val.size(0)
        epoch_val_loss /= len(val_dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # 早停机制
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发于 Epoch {epoch + 1}，验证Loss已收敛。")
                break

    # 恢复最佳权重并保存模型
    model.load_state_dict(best_model_state)
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "outputs" / "nn_demand_predictor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[M3] 最佳模型权重已保存至: {model_path.name}")

    # 绘制 Loss 曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss', marker='o', markersize=4, linewidth=1.5)
    ax.plot(val_losses, label='Validation Loss', marker='s', markersize=4, linewidth=1.5)
    ax.set_title('Neural Network Training & Validation Loss Curve', fontsize=13, pad=10)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    loss_plot_path = base_dir / "outputs" / "M3_nn_loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"[M3] Loss曲线已保存至: {loss_plot_path.name}")

    return {
        'model': model,
        'device': device,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_path': str(model_path),
        'loss_plot_path': str(loss_plot_path)
    }


def evaluate_and_compare_models(nn_model, device, X_train, y_train, X_test, y_test, features) -> dict:
    """
    NN测试集评估 + 随机森林基线对比 + 优劣分析报告
    """
    print("[M3] 正在执行模型评估与对比实验...")
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "outputs"

    # NN 测试集推理与评估
    nn_model.eval()
    with torch.no_grad():
        nn_preds = nn_model(torch.tensor(X_test).to(device)).cpu().numpy().flatten()
    nn_mae = mean_absolute_error(y_test, nn_preds)
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
    print(f"[M3] NN 测试集评估 | MAE: {nn_mae:.3f} | RMSE: {nn_rmse:.3f}")

    # 随机森林基线训练与评估
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    print(f"[M3]  RF 测试集评估 | MAE: {rf_mae:.3f} | RMSE: {rf_rmse:.3f}")

    # 生成对比分析报告
    better_model = "神经网络(NN)" if nn_rmse < rf_rmse else "随机森林(RF)"
    report_lines = [
        "=" * 60, "M3 预测模型对比实验报告", "=" * 60,
        f"测试集指标对比:",
        f"  • 神经网络(NN) | MAE: {nn_mae:.3f} | RMSE: {nn_rmse:.3f}",
        f"  • 随机森林(RF) | MAE: {rf_mae:.3f} | RMSE: {rf_rmse:.3f}",
        f"  • 综合表现更优: {better_model}",
        "",
        "算法优劣分析:",
        "神经网络 (NN) 优势：",
        "  1. 模型有能力自动发现区域、时段和历史需求等变量之间复杂的相互影响关系，不用人工去猜测或事先设定规则。",
        "  2. 训练时直接拿预测值与真实值的差距作为优化目标，再配合一种能自动调整步长的参数更新方式，让学习过程更加平稳，不易跑偏。",
        "  3. 这个框架很容易扩展：比如可以轻松加入专门处理“区域编号”这类类别信息的模块，或者换成更复杂的结构来捕捉更长的时间趋势。",
        "神经网络 (NN) 劣势：",
        "  1. 模型效果对参数设置（如学习快慢、每次喂入的数据量、网络层数）比较敏感，稍有不慎就容易过度记忆数据规律，"
        "  通常需要借助提前停止训练等技巧来防止这类问题，整体调起来比较麻烦。",
        "  2. 模型内部就像一个黑箱，很难直接看出各个因素到底起了多大作用，不方便追溯和解释具体原因，对交通业务的分析归因来说不太友好。",
        "随机森林 (RF) 优势：",
        "  1. 模型通过集成多棵树并取平均或投票，不容易受异常值和噪声干扰，在测试集上表现稳定。",
        "  2. 不需要对特征做标准化等预处理，使用默认参数通常就能得到不错的结果，训练也很快。",
        "  3. 能够直接输出每个特征的重要性，让人清楚地看到历史需求、时段等因素对预测结果的贡献有多大。",
        "随机森林 (RF) 劣势：",
        "  1. 模型高度依赖训练数据中的历史规律，对于超出历史范围的极端情况（如突发大型活动带来的需求暴涨），难以做出合理预判。",
        "  2. 树形结构在捕捉连续变量之间精细的渐变关系时能力有限，预测结果在细分的时空划分上容易出现不连贯的跳跃。",
        "=" * 60
    ]
    report_path = out_dir / "M3_model_comparison.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"[M3] 对比报告已保存至: {report_path.name}")

    return {
        'nn_mae': nn_mae, 'nn_rmse': nn_rmse,
        'rf_mae': rf_mae, 'rf_rmse': rf_rmse,
        'rf_model': rf,
        'report_path': str(report_path)
    }


def run_m3(df: pd.DataFrame) -> dict:
    """数据集 → NN训练 → 评估对比"""
    print("\n" + "=" * 50)
    print("启动 M3 预测模型模块")
    print("=" * 50)
    X_train, X_test, y_train, y_test, scaler, features = build_prediction_dataset(df)
    nn_res = train_nn_model(X_train, y_train, X_test, y_test)
    comp_res = evaluate_and_compare_models(nn_res['model'], nn_res['device'], X_train, y_train, X_test, y_test,
                                           features)
    print("M3 模块全量执行完毕 (NN训练+评估+RF对比)。\n")

    # 修复键名不匹配
    loss_curve_path = nn_res.get('loss_plot') or nn_res.get('loss_plot_path', '')

    return {
        'X_test': X_test, 'y_test': y_test, 'scaler': scaler, 'features': features,
        'nn_model': nn_res['model'], 'nn_device': nn_res['device'],
        'nn_paths': {'loss_curve': loss_curve_path},
        'metrics': {'nn_mae': comp_res['nn_mae'], 'nn_rmse': comp_res['nn_rmse'],
                    'rf_mae': comp_res['rf_mae'], 'rf_rmse': comp_res['rf_rmse']},
        'report_path': comp_res['report_path']
    }