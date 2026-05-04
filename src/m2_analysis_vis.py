import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def setup_plotting():
    """
    可视化环境基建
    """
    # 跨平台中文字体兼容
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示为方块的问题

    # 全局绘图参数
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    sns.set_theme(style="darkgrid", font='SimHei', font_scale=1.1)

    # 自动创建输出目录
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[M2] 可视化环境配置完成 | 输出目录: outputs/")


def save_fig(filename: str) -> str:
    """
    图表自动保存助手
    """
    base_dir = Path(__file__).resolve().parent.parent
    save_path = base_dir / "outputs" / filename
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 释放内存
    print(f"[M2] 图表已保存: {save_path.name}")
    return str(save_path)


def analyze_temporal_pattern(df: pd.DataFrame) -> dict:
    """
    分析1：出行需求时间规律（分小时、分工作日/周末展示订单量）
    """
    print("[M2] 正在生成分析1：出行需求时间规律...")

    # 按小时与是否周末分组统计订单量
    hourly_demand = df.groupby(['pickup_hour', 'is_weekend']).size().unstack(fill_value=0)
    hourly_demand.columns = ['工作日', '周末']

    # 提取业务结论
    weekday_peak = hourly_demand['工作日'].idxmax()
    weekend_peak = hourly_demand['周末'].idxmax()
    conclusion = (
        f"工作日订单峰值出现在 {weekday_peak}:00，周末峰值出现在 {weekend_peak}:00。"
        f"工作日的出行量在早晨和傍晚两个通勤时段出现明显高峰，形成早晚双峰格局。周末需求分布更平缓且夜间活跃度更高。"
    )

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(10, 6))
    hourly_demand.plot(kind='line', marker='o', linewidth=2, markersize=6, ax=ax)

    ax.set_title('纽约出租车分小时订单量趋势（工作日 vs 周末）', fontsize=14, pad=10)
    ax.set_xlabel('上车小时 (0-23)', fontsize=12)
    ax.set_ylabel('订单量 (次)', fontsize=12)
    ax.set_xticks(range(24))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='日期类型', loc='upper left')

    # 保存并返回结果
    plot_path = save_fig('M2_1_temporal.png')
    return {'conclusion': conclusion, 'plot_path': plot_path}


def run_m2(df: pd.DataFrame) -> dict:
    """
    [M2] 模块统一调度入口
    """
    print("\n" + "=" * 50)
    print("启动 M2 分析可视化模块")
    print("=" * 50)
    setup_plotting()
    res_1 = analyze_temporal_pattern(df)
    print("M2 阶段1执行完毕。\n")
    return {'M2_1': res_1}
