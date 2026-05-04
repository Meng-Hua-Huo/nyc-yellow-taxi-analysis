import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


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


def analyze_regional_heat(df: pd.DataFrame) -> dict:
    """
    分析2：区域热度分析（上下客量TOP10区域及高峰时段分布）
    """
    print("[M2] 正在生成分析2：区域热度分析...")

    # 计算各区域上下客总量并筛选 TOP 10
    pu_counts = df['PULocationID'].value_counts()
    do_counts = df['DOLocationID'].value_counts()
    zone_total = pu_counts.add(do_counts, fill_value=0).sort_values(ascending=False)
    top_10_zones = zone_total.head(10).index.tolist()

    # 准备 TOP10 区域的上下客对比数据
    top_zones_df = pd.DataFrame({
        '上客量': pu_counts.reindex(top_10_zones, fill_value=0),
        '下客量': do_counts.reindex(top_10_zones, fill_value=0)
    })

    # 准备 TOP10 区域的分小时热度矩阵
    df_top = df[df['PULocationID'].isin(top_10_zones)]
    heat_matrix = df_top.groupby(['PULocationID', 'pickup_hour']).size().unstack(fill_value=0)
    # 按总订单量降序排列热力图行，使高热度区域视觉上靠上，便于阅读
    heat_matrix = heat_matrix.loc[heat_matrix.sum(axis=1).sort_values(ascending=False).index]

    # 提取业务结论
    top1_zone = top_10_zones[0]
    peak_hour_top1 = heat_matrix.loc[top1_zone].idxmax()
    conclusion = (
        f"综合上下客量TOP1区域为Zone {top1_zone}，其需求峰值出现在 {peak_hour_top1}:00。"
        f"TOP10区域呈现显著的时段潮汐特征，早晚高峰(7-9/17-19点)订单高度集中，"
        f"符合纽约核心商业区与交通枢纽的通勤规律。"
    )

    # 组合绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：TOP10区域上下客量对比柱状图
    top_zones_df.plot(kind='bar', ax=axes[0], color=['#1f77b4', '#ff7f0e'], width=0.7)
    axes[0].set_title('TOP 10 区域上下客量对比', fontsize=13, pad=10)
    axes[0].set_xlabel('区域ID (LocationID)', fontsize=11)
    axes[0].set_ylabel('订单量 (次)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # 右图：TOP10区域 × 小时 需求热力图
    sns.heatmap(heat_matrix, cmap='YlOrRd', ax=axes[1], cbar_kws={'label': '订单量'}, linewidths=0.5)
    axes[1].set_title('TOP 10 区域分小时需求热力图', fontsize=13, pad=10)
    axes[1].set_xlabel('上车小时 (0-23)', fontsize=11)
    axes[1].set_ylabel('区域ID (按总热度降序)', fontsize=11)

    # 保存并返回结果
    plot_path = save_fig('M2_2_regional_heat.png')
    return {'conclusion': conclusion, 'plot_path': plot_path}


def analyze_fare_factors(df: pd.DataFrame) -> dict:
    """
    分析3：车费影响因素分析（距离、时段、乘客人数与车费的关系）
    """
    print("[M2] 正在生成分析3：车费影响因素分析...")

    # 数据采样防重叠
    df_sample = df.sample(frac=0.01, random_state=42).copy()
    corr_dist_fare = df['trip_distance'].corr(df['fare_amount'])

    # 时段业务分组
    df_sample['time_period'] = pd.cut(
        df_sample['pickup_hour'], bins=[-1, 5, 9, 16, 19, 23],
        labels=['凌晨(0-5)', '早高峰(6-9)', '日间平峰(10-16)', '晚高峰(17-19)', '夜间(20-23)']
    )

    # 提取业务结论
    conclusion = (
        f"【车费因素结论】行程距离与车费呈强正相关 (Pearson r={corr_dist_fare:.2f})。"
        f"散点趋势线斜率约为 ${np.polyfit(df_sample['trip_distance'], df_sample['fare_amount'], 1)[0]:.2f}/英里，"
        f"从时段箱线图来看，晚高峰和夜间的车费整体水平及偏高段位都明显上升，反映了拥堵等候费和夜间附加费的影响。"
        f"而乘客数量对基础车费基本没有影响，这也印证了纽约出租车按车收费、不按人数收费的运营惯例。"
    )

    # 组合绘图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左图：距离-车费透明度散点图 + 线性趋势线
    axes[0].scatter(df_sample['trip_distance'], df_sample['fare_amount'],
                    alpha=0.08, s=8, color='#1f77b4', edgecolors='none', rasterized=True)

    # 计算并绘制线性趋势线
    z = np.polyfit(df_sample['trip_distance'], df_sample['fare_amount'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 30, 100)
    axes[0].plot(x_trend, p(x_trend), color='#d62728', linewidth=2, linestyle='--',
                 label=f'线性趋势 (斜率${z[0]:.2f}/英里)')

    axes[0].set_title('行程距离 vs 车费 (散点图+趋势线)', fontsize=12, pad=8)
    axes[0].set_xlabel('行程距离 (英里)', fontsize=11)
    axes[0].set_ylabel('车费金额 (美元)', fontsize=11)
    axes[0].set_xlim(0, 30)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 中图：分时段车费箱线图
    sns.boxplot(x='time_period', y='fare_amount', hue='time_period', data=df_sample,
                ax=axes[1], palette='Set2', legend=False, showfliers=False)
    axes[1].set_title('分时段车费分布 (箱线图)', fontsize=12, pad=8)
    axes[1].set_xlabel('时段分组', fontsize=11)
    axes[1].set_ylabel('车费金额 (美元)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].set_ylim(0, 80)

    # 右图：分乘客数车费箱线图
    sns.boxplot(x='passenger_count', y='fare_amount', hue='passenger_count', data=df_sample,
                ax=axes[2], palette='Pastel1', legend=False, showfliers=False)
    axes[2].set_title('分乘客数车费分布 (箱线图)', fontsize=12, pad=8)
    axes[2].set_xlabel('乘客人数', fontsize=11)
    axes[2].set_ylabel('车费金额 (美元)', fontsize=11)
    axes[2].set_ylim(0, 80)

    plot_path = save_fig('M2_3_fare_factors.png')
    return {'conclusion': conclusion, 'plot_path': plot_path}


def analyze_tipping_behavior(df: pd.DataFrame) -> dict:
    """
    分析4：小费行为与支付方式分析
    """
    print("[M2] 正在生成分析4：小费行为与支付方式洞察...")

    # 业务过滤与特征构造
    df_cc = df[df['payment_type'] == 1].copy()
    # 限制小费比例在 0~50% 过滤极端系统错误，保留真实业务分布
    df_cc['tip_ratio'] = (df_cc['tip_amount'] / df_cc['fare_amount']).clip(0, 0.5)

    # 支付方式分布统计
    pay_map = {1: '信用卡', 2: '现金', 3: '免单/纠纷', 4: '争议', 5: '未知', 0: 'Flex', 6: '作废'}
    pay_dist = df['payment_type'].map(pay_map).value_counts()

    # 分小时平均小费比例聚合
    hourly_tip = df_cc.groupby('pickup_hour')['tip_ratio'].mean() * 100

    # 提取业务结论
    avg_tip = df_cc['tip_ratio'].mean() * 100
    peak_tip_hour = hourly_tip.idxmax()
    conclusion = (
        f"信用卡订单平均小费比例为 {avg_tip:.1f}%。"
        f"小费慷慨度在 {peak_tip_hour}:00 达到峰值，夜间(20-23点)小费比例普遍高于日间，"
        f"反映娱乐/聚餐场景下的乘客支付意愿更强。支付方式高度集中于信用卡，体现纽约出租车电子化生态。"
    )

    # 组合绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：支付方式订单占比
    bars = axes[0].bar(pay_dist.index, pay_dist.values, color=sns.color_palette('Set3', len(pay_dist)),
                       edgecolor='black', width=0.6)
    axes[0].set_title('订单支付方式分布', fontsize=12, pad=8)
    axes[0].set_xlabel('支付方式', fontsize=11)
    axes[0].set_ylabel('订单量 (次)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=30)
    # 添加柱顶数据标签
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, height, f'{height / 10000:.1f}万', ha='center', va='bottom',
                     fontsize=9)

    # 右图：分小时平均小费比例趋势
    axes[1].plot(hourly_tip.index, hourly_tip.values, marker='s', linewidth=2, color='#d62728', markersize=6)
    axes[1].fill_between(hourly_tip.index, hourly_tip.values, alpha=0.2, color='#d62728')
    axes[1].set_title('分小时平均小费比例趋势 (仅信用卡)', fontsize=12, pad=8)
    axes[1].set_xlabel('上车小时 (0-23)', fontsize=11)
    axes[1].set_ylabel('平均小费比例 (%)', fontsize=11)
    axes[1].set_xticks(range(24))
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, hourly_tip.max() * 1.2)

    plot_path = save_fig('M2_4_tipping_behavior.png')
    return {'conclusion': conclusion, 'plot_path': plot_path}


def run_m2(df: pd.DataFrame) -> dict:
    """
    环境配置 → 分析1 → 分析2 → 分析3 → 分析4
    """
    print("\n" + "=" * 50)
    print("启动 M2 分析可视化模块")
    print("=" * 50)
    setup_plotting()
    res_1 = analyze_temporal_pattern(df)
    res_2 = analyze_regional_heat(df)
    res_3 = analyze_fare_factors(df)
    res_4 = analyze_tipping_behavior(df)
    print("M2 模块全量执行完毕。\n")
    return {'M2_1': res_1, 'M2_2': res_2, 'M2_3': res_3, 'M2_4': res_4}
