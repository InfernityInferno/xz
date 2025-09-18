import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from vmdpy import VMD

# -------------------------- 1. 全局配置与状态初始化（关键：提前定义所有状态）
st.set_page_config(page_title="VMD-LSTM疫情预测系统", layout="wide")

# 用独立会话状态存储所有数据，避免分散导致的渲染冲突
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "file_processed": False,       # 标记文件是否已处理
        "model_loaded": False,         # 标记模型是否已加载
        "has_predictions": False,      # 标记是否有预测结果
        "processed_data": None,        # 处理后的数据
        "model": None,                 # 加载的模型
        "vmd_transform": None,         # VMD转换函数
        "predictions": None,           # 预测结果
        "start_date": None,            # 预测起始日期
        "end_date": None               # 预测结束日期
    }

# 模型固定参数（避免动态修改）
MODEL_PATH = "singlemodel(week).keras"
best_K = 9
window_size = 2


# -------------------------- 2. 核心功能函数（纯逻辑，不涉及Streamlit渲染）
def load_and_preprocess(uploaded_file):
    """纯数据处理函数，无Streamlit交互"""
    try:
        df = pd.read_csv(uploaded_file)
        if 'Date' not in df.columns or 'cases' not in df.columns:
            return {"success": False, "msg": "CSV需包含'Date'和'cases'列"}
        
        df['date'] = pd.to_datetime(df['Date'])
        df = df.set_index('date')
        data = df.resample('W-MON')['cases'].sum().dropna()
        
        if len(data) < window_size + 1:
            return {"success": False, "msg": f"需至少{window_size + 1}周数据"}
        
        # 标准化处理
        cases = data['cases'].values.reshape(-1, 1)
        train_end = int(0.7 * len(data))
        scaler = StandardScaler()
        scaler.fit(cases[:train_end])
        scaled_cases = scaler.transform(cases)
        
        return {
            "success": True,
            "data": {
                "raw_data": data,
                "scaled_cases": scaled_cases,
                "scaler": scaler,
                "last_monday": data.index[-1].date()
            }
        }
    except Exception as e:
        return {"success": False, "msg": f"数据处理失败：{str(e)}"}


def init_model_and_vmd(scaled_cases):
    """纯模型初始化函数，无Streamlit交互"""
    if not os.path.exists(MODEL_PATH):
        return {"success": False, "msg": f"未找到模型文件：{MODEL_PATH}"}
    
    try:
        model = load_model(MODEL_PATH)
        train_end = int(0.7 * len(scaled_cases))
        
        # 定义VMD转换（固定参数）
        def vmd_transform(signal):
            alpha = 5000
            tau = 0.1
            DC = 0
            init = 1
            tol = 1e-7
            imfs, _, _ = VMD(signal.flatten(), alpha, tau, best_K, DC, init, tol)
            return np.array(imfs).T
        
        # 用训练集初始化VMD（仅执行一次）
        vmd_transform(scaled_cases[:train_end])
        return {
            "success": True,
            "model": model,
            "vmd_transform": vmd_transform
        }
    except Exception as e:
        return {"success": False, "msg": f"模型加载失败：{str(e)}"}


def generate_predictions(model, vmd_transform, processed_data, start_date, end_date):
    """纯预测函数，无Streamlit交互"""
    last_monday = processed_data['last_monday']
    scaled_cases = processed_data['scaled_cases']
    scaler = processed_data['scaler']
    
    # 计算周数差
    start_diff = (start_date - last_monday).days // 7
    end_diff = (end_date - last_monday).days // 7
    
    if start_diff <= 0 or end_diff <= 0 or start_date > end_date:
        return {"success": False, "msg": "起始周需在结束周前，且均晚于最后数据周"}
    
    # 预测逻辑
    temp_cases = scaled_cases[-window_size:].copy()
    current_imfs = vmd_transform(temp_cases)
    
    # 先预测到起始周
    for _ in range(start_diff):
        pred_scaled = model.predict(current_imfs.reshape(1, window_size, best_K), verbose=0)[0][0]
        temp_cases = np.append(temp_cases[1:], pred_scaled).reshape(-1, 1)
        current_imfs = vmd_transform(temp_cases)[-window_size:]
    
    # 预测目标周
    predictions = []
    current_date = start_date
    total_weeks = end_diff - start_diff + 1
    
    for _ in range(total_weeks):
        pred_scaled = model.predict(current_imfs.reshape(1, window_size, best_K), verbose=0)[0][0]
        pred_actual = round(scaler.inverse_transform([[pred_scaled]])[0][0])
        week_end = current_date + timedelta(days=6)
        predictions.append({
            "start_date": current_date,
            "end_date": week_end,
            "prediction": pred_actual
        })
        
        # 更新窗口
        temp_cases = np.append(temp_cases[1:], pred_scaled).reshape(-1, 1)
        current_imfs = vmd_transform(temp_cases)[-window_size:]
        current_date += timedelta(weeks=1)
    
    return {"success": True, "predictions": predictions}


# -------------------------- 3. Streamlit界面（严格分区，减少动态嵌套）
def main():
    # 固定标题（无动态变化）
    st.title("VMD-LSTM疫情周预测系统")
    
    # -------------------------- 3.1 第一区：文件上传（仅处理一次）
    st.subheader("1. 上传数据文件")
    uploaded_file = st.file_uploader("选择CSV文件（含'Date'和'cases'列）", type="csv", key="file_upload")
    
    # 仅在文件上传且未处理过时执行
    if uploaded_file and not st.session_state.app_state["file_processed"]:
        with st.spinner("处理数据中..."):
            result = load_and_preprocess(uploaded_file)
            if result["success"]:
                st.session_state.app_state["processed_data"] = result["data"]
                st.session_state.app_state["file_processed"] = True
                st.success("数据处理完成！")
            else:
                st.error(result["msg"])
    
    # -------------------------- 3.2 第二区：数据概览（仅在文件处理后显示）
    if st.session_state.app_state["file_processed"]:
        st.subheader("2. 数据概览")
        data = st.session_state.app_state["processed_data"]["raw_data"]
        last_monday = st.session_state.app_state["processed_data"]["last_monday"]
        
        st.write(f"数据范围：{data.index.min().strftime('%Y-%m-%d')}（周一）至 {last_monday.strftime('%Y-%m-%d')}（周一）")
        st.write(f"总周数：{len(data)}周")
        st.dataframe(data.head(5), use_container_width=True)
        
        # -------------------------- 3.3 第三区：模型加载（仅在数据处理后执行一次）
        if not st.session_state.app_state["model_loaded"]:
            with st.spinner("加载模型中..."):
                scaled_cases = st.session_state.app_state["processed_data"]["scaled_cases"]
                model_result = init_model_and_vmd(scaled_cases)
                if model_result["success"]:
                    st.session_state.app_state["model"] = model_result["model"]
                    st.session_state.app_state["vmd_transform"] = model_result["vmd_transform"]
                    st.session_state.app_state["model_loaded"] = True
                    st.success("模型加载完成！")
                else:
                    st.error(model_result["msg"])
        
        # -------------------------- 3.4 第四区：预测配置（仅在模型加载后显示）
        if st.session_state.app_state["model_loaded"]:
            st.subheader("3. 预测配置")
            last_monday = st.session_state.app_state["processed_data"]["last_monday"]
            
            # 日期选择器（固定初始值，避免动态变化）
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "起始周（周一）",
                    value=last_monday + timedelta(weeks=1),
                    min_value=last_monday + timedelta(weeks=1),
                    format="YYYY-MM-DD",
                    key="start_picker"
                )
            with col2:
                end_date = st.date_input(
                    "结束周（周一）",
                    value=last_monday + timedelta(weeks=3),
                    min_value=start_date,
                    format="YYYY-MM-DD",
                    key="end_picker"
                )
            
            # 日期验证（提前判断，避免点击后报错）
            date_valid = True
            if start_date.weekday() != 0:
                st.warning("起始周必须是周一")
                date_valid = False
            if end_date.weekday() != 0:
                st.warning("结束周必须是周一")
                date_valid = False
            
            # 预测按钮（仅在日期有效时可点击）
            if date_valid and st.button("生成预测", key="predict_btn"):
                with st.spinner("预测中..."):
                    pred_result = generate_predictions(
                        model=st.session_state.app_state["model"],
                        vmd_transform=st.session_state.app_state["vmd_transform"],
                        processed_data=st.session_state.app_state["processed_data"],
                        start_date=start_date,
                        end_date=end_date
                    )
                    if pred_result["success"]:
                        st.session_state.app_state["predictions"] = pred_result["predictions"]
                        st.session_state.app_state["has_predictions"] = True
                        st.session_state.app_state["start_date"] = start_date
                        st.session_state.app_state["end_date"] = end_date
                    else:
                        st.error(pred_result["msg"])
        
        # -------------------------- 3.5 第五区：预测结果（独立容器，仅在有结果时显示）
        if st.session_state.app_state["has_predictions"]:
            st.subheader("4. 预测结果")
            predictions = st.session_state.app_state["predictions"]
            start_date = st.session_state.app_state["start_date"]
            end_date = st.session_state.app_state["end_date"]
            
            # 结果表格（用DataFrame，避免Table渲染冲突）
            result_df = pd.DataFrame([
                {
                    "周范围": f"{p['start_date'].strftime('%Y-%m-%d')} 至 {p['end_date'].strftime('%Y-%m-%d')}",
                    "预测病例数": p["prediction"]
                } for p in predictions
            ])
            st.dataframe(result_df, use_container_width=True)
            
            # 总病例数
            total_cases = sum(p["prediction"] for p in predictions)
            st.markdown(f"### 预测总病例数：{total_cases} 例")
            
            # 重置按钮（单独一行，避免嵌套）
            if st.button("重置预测结果", key="reset_btn"):
                st.session_state.app_state["has_predictions"] = False
                st.session_state.app_state["predictions"] = None
                st.experimental_rerun()  # 强制刷新，重置DOM状态


# -------------------------- 4. 侧边栏（纯静态信息，无动态交互）
with st.sidebar:
    st.header("模型参数")
    st.write(f"VMD分解K值：{best_K}")
    st.write(f"滑动窗口大小：{window_size}")
    st.write(f"LSTM单元数：32")
    st.write(f"标准化方式：StandardScaler")
    st.write(f"周起始日：周一")
    st.info("使用步骤：\n1. 上传CSV文件\n2. 等待数据和模型加载\n3. 选择预测周（均为周一）\n4. 点击生成预测")


# 启动应用
if __name__ == "__main__":
    main()
