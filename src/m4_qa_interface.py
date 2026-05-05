import re
import os
import requests
import numpy as np
import torch
import pandas as pd
from pathlib import Path

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
                print("[系统提示] 问题未命中规则，转由大模型分析...")
                llm_reply = call_llm_fallback(query)
                # 格式化输出大模型回复
                print("\n" + "-" * 40)
                print(llm_reply)
                print("-" * 40)
                continue

            # 路由调用与异常兜底
            handler = INTENT_HANDLERS.get(intent)
            try:
                if intent == 'predict':
                    res = handler(query, m3_data)
                else:
                    res = handler(m2_results)
            except Exception as e:
                # 规则处理失败，降级到大模型
                print(f"[规则处理异常] {e}，尝试由大模型回答...")
                llm_reply = call_llm_fallback(query)
                print("\n" + "-" * 40)
                print(llm_reply)
                print("-" * 40)
                continue

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


LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY"),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/"),
    "model": os.getenv("LLM_MODEL", "deepseek-v4-flash"),
    "timeout": 15
}


def get_system_prompt() -> str:
    """
    设计与迭代记录
    ----------------------------------------------------------------
    v1: 你是一个出租车数据分析助手。请回答用户问题。
      问题：回复发散，易编造数据，未限定能力边界。
    v2: 增加数据范围(2023年1月纽约黄车)、可用模块列表、要求结构化输出。
      问题：遇到超纲问题仍会强行推理，缺乏明确的“拒绝编造”指令。
    v3(当前): 严格划定知识边界；明确“未知即拒绝”原则；
      规定低温度 (0.3)与客观语气。彻底解决大模型在垂直数据问答中的幻觉问题。
    """
    return """你是一个专业的纽约市出租车出行数据分析助手。
你的知识严格限于2023年1月纽约黄色出租车公开数据集。切勿编造数据、日期或统计结果。
当前系统已内置以下分析模块：
1. 时间规律分析（分小时/工作日周末订单趋势）
2. 区域热度分析（TOP10上下客区域及高峰时段）
3. 车费影响因素（距离/时段/乘客数与车费关系）
4. 小费与支付行为分析
5. 区域时段需求预测（基于神经网络）
回复规则:
- 若用户问题超出数据范围或系统能力，请明确告知“当前数据/模型暂不支持该分析”，并给出替代建议。绝对不要编造数值。
- 若问题属于上述5类但表述模糊，请引导用户补充关键参数（如具体区域ID、时段）。
- 回复需简洁、专业、结构化。优先使用要点列表。
- 语气保持客观、数据驱动。
示例:
用户: "为什么下雨天打车这么贵？"
助手: "当前数据集(2023年1月)未包含天气字段，因此无法直接分析降雨对车费的影响。建议您关注‘时段’与‘区域’维度，系统已验证晚高峰(17-19点)与核心商业区存在显著溢价现象。如需预测特定区域需求，可提供Zone ID与时间。"
请严格遵循以上规则回复用户问题。"""


def call_llm_fallback(user_query: str) -> str:
    """调用大模型API进行兜底回复 (兼容 GLM/DeepSeek/Qwen OpenAI格式)"""
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_CONFIG["model"],
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.3,  # 低温度保证回复稳定、减少幻觉
        "max_tokens": 600
    }
    try:
        response = requests.post(
            f"{LLM_CONFIG['base_url']}chat/completions",
            headers=headers, json=payload, timeout=LLM_CONFIG["timeout"]
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"【LLM调用失败】网络或API请求异常: {str(e)}。请检查API Key或网络连接。"
    except Exception as e:
        return f"【LLM解析失败】模型返回格式异常: {str(e)}"


def run_m4(m2_results: dict, m3_data: dict):
    """模块统一调度入口"""
    print("\n" + "=" * 50)
    print("启动 M4 智能问答接口模块")
    print("=" * 50)
    run_qa_loop(m2_results, m3_data)
    print("M4 模块执行完毕。\n")
