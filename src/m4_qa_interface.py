import re
import numpy as np
import torch
from pathlib import Path
import pandas as pd

INTENT_PATTERNS = {
    'temporal': r'(时间|规律|时段|小时|工作日|周末|订单量|骑行量|趋势)',
    'regional': r'(区域|热度|排名|top|上下客|地点|哪里|热门)',
    'fare': r'(车费|费用|价格|距离|乘客|多少钱|计价|因素)',
    'tipping': r'(小费|支付|方式|信用卡|现金|慷慨|打赏)',
    'predict': r'(预测|需求|未来|估计|多少单|单量|需求量)'
}


def match_intent(query: str) -> str:
    """规则意图识别"""
    query = query.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, query):
            return intent
    return 'unknown'


def extract_predict_params(query: str) -> dict:
    """预测参数提取（区域ID、小时、是否周末）"""
    # 提取区域ID
    zone_match = re.search(r'(\d{1,3})\s*(区|zone|区域)', query)
    zone_id = int(zone_match.group(1)) if zone_match else 161

    # 提取小时
    hour_match = re.search(r'(\d{1,2})\s*[点时]', query)
    hour = int(hour_match.group(1)) if hour_match else 18
    hour = max(0, min(23, hour))  # 边界保护

    # 提取日期类型
    is_weekend = 1 if re.search(r'(周末|周六|周日|休息日)', query) else 0
    dayofweek = 6 if is_weekend else 2  # 6=周日, 2=周三(代表工作日)

    return {'zone_id': zone_id, 'hour': hour, 'is_weekend': is_weekend, 'dayofweek': dayofweek}


def handle_temporal(m2_results: dict) -> dict:
    return m2_results.get('M2_1', {'conclusion': '未找到时间规律分析结果。', 'plot_path': ''})


def handle_regional(m2_results: dict) -> dict:
    return m2_results.get('M2_2', {'conclusion': '未找到区域热度分析结果。', 'plot_path': ''})


def handle_fare(m2_results: dict) -> dict:
    return m2_results.get('M2_3', {'conclusion': '未找到车费因素分析结果。', 'plot_path': ''})


def handle_tipping(m2_results: dict) -> dict:
    return m2_results.get('M2_4', {'conclusion': '未找到小费行为分析结果。', 'plot_path': ''})


def handle_predict(query: str, m3_data: dict) -> dict:
    """需求预测推理（调用M3训练好的NN模型）"""
    try:
        params = extract_predict_params(query)
        zone, hour, is_wknd, dow = params['zone_id'], params['hour'], params['is_weekend'], params['dayofweek']
        is_peak = 1 if hour in [7, 8, 9, 17, 18, 19] else 0

        # 构造特征向量
        lag1, lag24 = 12.0, 12.0
        feat_raw = np.array([[zone, hour, dow, is_wknd, is_peak, lag1, lag24]], dtype=np.float32)

        # 仅对数值列标准化
        scaler = m3_data['scaler']
        num_cols_idx = [1, 2, 5, 6]
        num_col_names = ['pickup_hour', 'dayofweek', 'lag_1h', 'lag_24h']

        df_to_scale = pd.DataFrame(feat_raw[:, num_cols_idx], columns=num_col_names)
        feat_raw[:, num_cols_idx] = scaler.transform(df_to_scale)

        # 模型推理
        model = m3_data['nn_model']
        device = m3_data['nn_device']
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(feat_raw).to(device)).cpu().item()
        pred = max(0.0, pred)  # 需求不能为负

        conclusion = (
            f"需求预测结论：基于神经网络模型预测：\n"
            f"区域 Zone {zone} 在 {'周末' if is_wknd else '工作日'} {hour}:00 的预计订单量为 {pred:.1f} 单。\n"
            f"注：预测基于历史平均滞后需求(12单)推算，实际值受天气/活动/路况影响会有波动。"
        )
        return {'conclusion': conclusion, 'plot_path': m3_data['nn_paths'].get('loss_curve', '')}
    except Exception as e:
        return {'conclusion': f'【预测失败】模型推理异常: {str(e)}', 'plot_path': ''}


INTENT_HANDLERS = {
    'temporal': handle_temporal,
    'regional': handle_regional,
    'fare': handle_fare,
    'tipping': handle_tipping,
    'predict': handle_predict
}


def run_qa_loop(m2_results: dict, m3_data: dict):
    """
    命令行问答循环
    """
    print("\n" + "=" * 60)
    print("欢迎使用 纽约出租车出行数据智能问答系统 (M4)")
    print("支持提问类型: 时间规律 | 区域热度 | 车费因素 | 小费支付 | 需求预测")
    print("输入 exit / quit / 退出 可结束问答。按 Ctrl+C 也可安全退出。")
    print("=" * 60)

    while True:
        try:
            query = input("\n>> 请输入您的问题: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit', '退出', 'q']:
                print("感谢使用，系统已安全退出。")
                break

            # 意图识别
            intent = match_intent(query)
            if intent == 'unknown':
                print("【系统提示】暂未识别到您的意图。请尝试包含关键词：时间/区域/车费/小费/预测。")
                continue

            # 路由调用与异常兜底
            handler = INTENT_HANDLERS.get(intent)
            try:
                if intent == 'predict':
                    res = handler(query, m3_data)
                else:
                    res = handler(m2_results)
            except Exception as e:
                res = {'conclusion': f'【系统错误】处理请求时发生异常: {str(e)}', 'plot_path': ''}

            # 结果格式化输出
            print("\n" + "-" * 40)
            print(res['conclusion'])
            if res.get('plot_path') and Path(res['plot_path']).exists():
                print(f"【图表】已保存至: {Path(res['plot_path']).name}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n检测到中断信号，系统已安全退出。")
            break
        except EOFError:
            break


def run_m4(m2_results: dict, m3_data: dict):
    """模块统一调度入口"""
    print("\n" + "=" * 50)
    print("启动 M4 智能问答接口模块")
    print("=" * 50)
    run_qa_loop(m2_results, m3_data)
    print("M4 模块执行完毕。\n")
