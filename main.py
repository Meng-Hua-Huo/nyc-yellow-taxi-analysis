import sys
from pathlib import Path

# 确保能导入内部模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from m1_data_processing import load_data, generate_quality_report, clean_data, run_m1

def main():
    print("启动 AI出租车数据分析系统 | M1 数据处理")
    try:
        df_raw = load_data()
        report_path = generate_quality_report(df_raw)
        df_final = run_m1()
        print("清洗后数据预览:")
        print(df_final.head())
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
