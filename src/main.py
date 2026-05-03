import traceback
from src.m1_data_processing import load_data

def main():
    print("===== 纽约出租车数据分析系统 =====")
    try:
        print("正在加载数据...")
        df = load_data()
        print("\n前5行预览:")
        print(df.head())
        print("\n数据信息:")
        df.info()
    except Exception as e:
        print(f"\n运行失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()