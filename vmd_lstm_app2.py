import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from vmdpy import VMD

# 页面配置与全局设置
st.set_page_config(page_title="VMD-LSTM疫情预测系统", layout="wide")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 模型路径和关键参数
MODEL_PATH = "singlemodel(week).keras"  # 模型文件路径
best_K = 9  # 最佳VMD分解K值
window_size = 2  # 滑动窗口大小

# 数据加载与预处理（从上传CSV文件读取）
def load_and_preprocess_uploaded_data(uploaded_file):
    try:
        # 读取上传的CSV文件
        df = pd.read_csv(uploaded_file)
        
        # 检查必要的列是否存在
        if 'Date' not in df.columns:
            st.error("上传的CSV文件必须包含'Date'列")
            return None
        if 'cases' not in df.columns:
            st.error("上传的CSV文件必须包含'cases'列")
            return None
        
        # 处理日期列
        df['date'] = pd.to_datetime(df['Date'])
        
        # 按周聚合（周一为起始）
        df = df.set_index('date')
        data = df.resample('W-MON').agg({
            'cases': 'sum'  # 只聚合病例数据
        }).dropna()
        
        if len(data) < window_size + 1:
            st.error(f"数据量不足，至少需要{window_size + 1}周的数据")
            return None
        
        # 提取病例数据
        cases = data['cases'].values.reshape(-1, 1)
        
        # 计算训练集边界（用于拟合标准化器）
        total_len = len(data)
        train_end = int(0.7 * total_len)
        
        # 拟合标准化器
        case_scaler = StandardScaler()
        case_scaler.fit(cases[:train_end])
        
        # 标准化数据
        scaled_cases = case_scaler.transform(cases)
        
        return {
            'data': data,
            'scaled_cases': scaled_cases,
            'case_scaler': case_scaler,
            'last_monday': data.index[-1].date()
        }
    except Exception as e:
        st.error(f"数据处理失败：{str(e)}")
        return None

# VMD分解函数
def fit_vmd(train_signal, K):
    alpha = 5000
    tau = 0.1
    DC = 0
    init = 1
    tol = 1e-7
    
    def transform(signal):
        imfs, _, _ = VMD(
            signal.flatten(), 
            alpha=alpha, 
            tau=tau, 
            K=K, 
            DC=DC, 
            init=init, 
            tol=tol
        )
        return np.array(imfs).T  # 形状: (n_samples, K)
    
    return transform

# 加载模型与初始化VMD
def initialize_model_and_vmd(scaled_cases):
    if not os.path.exists(MODEL_PATH):
        st.error(f"未找到模型文件：{MODEL_PATH}")
        return None, None
    
    try:
        # 加载模型
        model = load_model(MODEL_PATH)
        
        # 用训练集数据初始化VMD
        total_len = len(scaled_cases)
        train_end = int(0.7 * total_len)
        vmd_transform = fit_vmd(scaled_cases[:train_end], K=best_K)
        
        return model, vmd_transform
    except Exception as e:
        st.error(f"模型加载失败：{str(e)}")
        return None, None

# 单周预测函数
def predict_single_week(model, vmd_transform, current_imfs, case_scaler, window_size, best_K):
    X_pred = current_imfs.reshape(1, window_size, best_K)
    pred_scaled = model.predict(X_pred, verbose=0)[0][0]
    return pred_scaled, case_scaler.inverse_transform([[pred_scaled]])[0][0]

# 多周预测函数
def predict_multiple_weeks(model, vmd_transform, processed_data, start_date, end_date):
    last_monday = processed_data['last_monday']
    scaled_cases = processed_data['scaled_cases']
    case_scaler = processed_data['case_scaler']
    
    # 计算预测周数差
    start_weeks_diff = (start_date - last_monday).days // 7
    end_weeks_diff = (end_date - last_monday).days // 7
    
    if start_weeks_diff <= 0 or end_weeks_diff <= 0 or start_date > end_date:
        st.warning("请确保起始周在结束周之前，且都晚于最后一个数据点的日期")
        return None
    
    # 用最近的窗口数据初始化
    temp_cases = scaled_cases[-window_size:].copy()
    current_imfs = vmd_transform(temp_cases)
    
    # 先预测到起始周
    for _ in range(start_weeks_diff):
        pred_scaled, _ = predict_single_week(model, vmd_transform, current_imfs, case_scaler, window_size, best_K)
        temp_cases = np.append(temp_cases[1:], pred_scaled).reshape(-1, 1)
        current_imfs = vmd_transform(temp_cases)[-window_size:]
    
    # 预测起始周到结束周
    current_date = start_date
    total_weeks = end_weeks_diff - start_weeks_diff + 1
    predictions = []
    
    for _ in range(total_weeks):
        pred_scaled, pred_actual = predict_single_week(model, vmd_transform, current_imfs, case_scaler, window_size, best_K)
        week_end = current_date + timedelta(days=6)
        predictions.append({
            'start_date': current_date,
            'end_date': week_end,
            'prediction': round(pred_actual)
        })
        
        # 更新窗口
        temp_cases = np.append(temp_cases[1:], pred_scaled).reshape(-1, 1)
        current_imfs = vmd_transform(temp_cases)[-window_size:]
        current_date += timedelta(weeks=1)
    
    return predictions

# Streamlit界面
st.title("VMD-LSTM疫情周预测系统（支持CSV上传）")

# 文件上传部分
st.subheader("1. 上传数据文件")
uploaded_file = st.file_uploader("请上传包含'Date'和'cases'列的CSV文件", type=["csv"])

if uploaded_file is not None:
    # 处理上传的数据
    processed_data = load_and_preprocess_uploaded_data(uploaded_file)
    
    if processed_data is not None:
        # 显示数据信息
        st.subheader("2. 数据概览")
        last_monday = processed_data['last_monday']
        data = processed_data['data']
        st.write(f"数据时间范围：{data.index.min().strftime('%Y-%m-%d')}（周一）至 {last_monday.strftime('%Y-%m-%d')}（周一）")
        st.write(f"总周数：{len(data)}周")
        st.info("数据已按每周一至周日聚合，索引日期为每周一")
        
        # 显示前5行数据预览
        st.write("数据预览（前5行）：")
        st.dataframe(data.head())
        
        # 加载模型和VMD
        model, vmd_transform = initialize_model_and_vmd(processed_data['scaled_cases'])
        
        # 预测部分
        if model is not None and vmd_transform is not None:
            st.subheader("3. 未来预测")
            
            # 日期选择器（基于最后一个周一）
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "选择起始周（周一）",
                    value=last_monday + timedelta(weeks=1),
                    min_value=last_monday + timedelta(weeks=1),
                    format="YYYY-MM-DD"
                )
            
            with col2:
                end_date = st.date_input(
                    "选择结束周（周一）",
                    value=last_monday + timedelta(weeks=3),
                    min_value=start_date,
                    format="YYYY-MM-DD"
                )
            
            # 验证日期是否为周一
            date_valid = True
            if start_date.weekday() != 0:
                st.warning("起始周请选择周一")
                date_valid = False
            if end_date.weekday() != 0:
                st.warning("结束周请选择周一")
                date_valid = False
            
            if date_valid:
                week_count = ((end_date - start_date).days // 7) + 1
                st.info(f"共预测 {week_count} 周数据（从 {start_date} 到 {end_date}）")
                
                if st.button("生成预测"):
                    with st.spinner("正在预测..."):
                        predictions = predict_multiple_weeks(
                            model, 
                            vmd_transform, 
                            processed_data, 
                            start_date, 
                            end_date
                        )
                        
                        if predictions is not None:
                            st.success("预测完成！")
                            
                            # 显示结果表格
                            result_data = []
                            total = 0
                            for item in predictions:
                                week_range = f"{item['start_date'].strftime('%Y-%m-%d')} 至 {item['end_date'].strftime('%Y-%m-%d')}"
                                result_data.append({"周范围": week_range, "预测病例数": item['prediction']})
                                total += item['prediction']
                            
                            st.table(pd.DataFrame(result_data))
                            st.markdown(f"### 预测总病例数：{total} 例")
                            
# 侧边栏参数
with st.sidebar:
    st.header("模型参数")
    st.write(f"最佳VMD分解K值：{best_K}")
    st.write(f"滑动窗口大小：{window_size}")
    st.write(f"LSTM单元数：32")
    st.write(f"数据标准化方式：StandardScaler")
    st.write(f"周起始日：周一")
    st.info("使用说明：\n1. 上传包含'Date'和'cases'列的CSV文件\n2. 系统会自动按周聚合数据\n3. 选择预测的起始和结束周（均为周一）\n4. 点击生成预测按钮查看结果")
    
