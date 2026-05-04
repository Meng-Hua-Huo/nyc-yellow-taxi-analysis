import sys
from pathlib import Path

# 将 src/ 加入模块搜索路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from m1_data_processing import run_m1
from m2_analysis_vis import run_m2


def main():
    print("启动 AI出租车数据分析系统 | 全流程入口")
    try:
        # 执行 M1 数据处理
        df_clean = run_m1()

        # 执行 M2 分析可视化
        m2_results = run_m2(df_clean)

        # 打印阶段结果摘要
        print("\n" + "=" * 50)
        print("当前阶段输出摘要:")
        for k, v in m2_results.items():
            print(f" {k}: {v['conclusion']}")
            print(f" 图表路径: {v['plot_path']}")
        print("=" * 50)

    except Exception as e:
        print(f"\n 流程中断: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
