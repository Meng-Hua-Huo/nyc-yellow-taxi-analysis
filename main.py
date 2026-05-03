import sys
from pathlib import Path

# 确保能导入内部模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from m1_data_processing import load_data, generate_quality_report

def main():
    print("启动 AI出租车数据分析系统 | M1 数据处理")
    try:
        df_raw = load_data()
        report_path = generate_quality_report(df_raw)
        print(f"M1 阶段完成。报告路径: {report_path}")
    except Exception as e:
        print(f"流程中断: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
