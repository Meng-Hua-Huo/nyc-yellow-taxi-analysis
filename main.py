import sys
from pathlib import Path

# 确保能导入内部模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from m1_data_processing import load_data, generate_quality_report, clean_data, run_m1, extract_time_features
from m1_data_processing import create_derived_features, save_cleaned_data
def main():
    print("启动 AI出租车数据分析系统 | M1 数据处理")
    try:
        df_final = run_m1()
        print("清洗后数据预览:")
        print(df_final.head())
        print("\n时间特征列预览:")
        print(df_final[['tpep_pickup_datetime', 'pickup_hour', 'pickup_dayofweek', 'is_weekend', 'is_peak_hour']].head(
            10))
        print("\n特征分布统计:")
        print(df_final[['pickup_hour', 'is_weekend', 'is_peak_hour']].describe())
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
