import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from vmdpy import VMD

# 全局配置与会话状态初始化
st.set_page_config(page_title="VMD-LSTM疫情预测系统", layout="wide")

if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        "file_processed": False,
        "model_loaded": False,
        "has_predictions": False,
        "processed_data": None,
        "model": None,
        "vmd_transform": None,
        "predictions": None,
        "start_date": None,
        "end_date": None
    }

# 模型固定参数
MODEL_PATH = "singlemodel(week).keras"
best_K = 9
window_size = 2


# 核心功能函数
def load_and_preprocess(uploaded_file):
    """读取CSV文件并预处理"""
    try:
        df = pd.read_csv(uploaded_file)
        
        required_columns = ['Date', 'cases']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return {"success": False, "msg": f"CSV需包含以下列：{', '.join(missing_cols)}"}
        
        df['date'] = pd.to_datetime(df['Date'])
        df = df.set_index('date')
        data = df.resample('W-MON').agg({'cases': 'sum'}).dropna()
        
        if len(data) < window_size + 1:
            return {"success": False, "msg": f"需至少{window_size + 1}周数据（当前{len(data)}周）"}
        
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
    """初始化模型和VMD转换"""
    if not os.path.exists(MODEL_PATH):
        return {"success": False, "msg": f"未找到模型文件：{MODEL_PATH}"}
    
    try:
        model = load_model(MODEL_PATH)
        train_end = int(0.7 * len(scaled_cases))
        
        def vmd_transform(signal):
            alpha = 5000
            tau = 0.1
            DC = 0
            init = 1
            tol = 1e-7
            imfs, _, _ = VMD(signal.flatten(), alpha, tau, best_K, DC, init, tol)
            return np.array(imfs).T
        
        vmd_transform(scaled_cases[:train_end])
        return {
            "success": True,
            "model": model,
            "vmd_transform": vmd_transform
        }
    except Exception as e:
        return {"success": False, "msg": f"模型加载失败：{str(e)}"}


def predict_single_week(model, vmd_transform, current_imfs, scaler):
    """单周预测"""
    X_pred = current_imfs.reshape(1, window_size, best_K)
    pred_scaled = model.predict(X_pred, verbose=0)[0][0]
    pred_actual = round(scaler.inverse_transform([[pred_scaled]])[0][0])
    return pred_scaled, pred_actual


def generate_predictions(model, vmd_transform, processed_data, start_date, end_date):
    """多周预测"""
    last_monday = processed_data['last_monday']
    scaled_cases = processed_data['scaled_cases']
    scaler = processed_data['scaler']
    
    start_diff = (start_date - last_monday).days // 7
    end_diff = (end_date - last_monday).days // 7
    
    if start_diff <= 0 or end_diff <= 0 or start_date > end_date:
        return {"success": False, "msg": "起始周/结束周需晚于最后数据周，且起始周≤结束周"}
    
    temp_cases = scaled_cases[-window_size:].copy()
    current_imfs = vmd_transform(temp_cases)
    
    for _ in range(start_diff):
        pred_scaled, _ = predict_single_week(model, vmd_transform, current_imfs, scaler)
        temp_cases = np.append(temp_cases[1:], pred_scaled).reshape(-1, 1)
        current_imfs = vmd_transform(temp_cases)[-window_size:]
    
    predictions = []
    current_date = start_date
    total_weeks = end_diff - start_diff + 1
    
    for _ in range(total_weeks):
        pred_scaled, pred_actual = predict_single_week(model, vmd_transform, current_imfs, scaler)
        week_end = current_date + timedelta(days=6)
        predictions.append({
            "start_date": current_date,
            "end_date": week_end,
            "prediction": pred_actual
        })
        
        temp_cases = np.append(temp_cases[1:], pred_scaled).reshape(-1, 1)
        current_imfs = vmd_transform(temp_cases)[-window_size:]
        current_date += timedelta(weeks=1)
    
    return {"success": True, "predictions": predictions}


# Streamlit界面
def main():
    st.title("VMD-LSTM疫情周预测系统（CSV上传版）")
    
    # 文件上传
    st.subheader("1. 上传CSV数据文件")
    uploaded_file = st.file_uploader(
        "选择CSV文件（需包含'Date'和'cases'列）",
        type=["csv"],
        key="file_upload"
    )
    
    if uploaded_file and not st.session_state.app_state["file_processed"]:
        with st.spinner("处理CSV数据中..."):
            result = load_and_preprocess(uploaded_file)
            if result["success"]:
                st.session_state.app_state["processed_data"] = result["data"]
                st.session_state.app_state["file_processed"] = True
                st.success("✅ CSV数据处理完成！")
            else:
                st.error(f"❌ {result['msg']}")
    
    # 数据概览
    if st.session_state.app_state["file_processed"]:
        st.subheader("2. 数据概览")
        data = st.session_state.app_state["processed_data"]["raw_data"]
        last_monday = st.session_state.app_state["processed_data"]["last_monday"]
        
        st.write(f"📅 数据时间范围：{data.index.min().strftime('%Y-%m-%d')}（周一）至 {last_monday.strftime('%Y-%m-%d')}（周一）")
        st.write(f"📊 总周数：{len(data)}周（已按“周一至周日”聚合）")
        
        # 修复：用单独的caption函数替代参数
        st.caption("数据预览（前5行）")
        st.dataframe(data.head(5), use_container_width=True)
        
        # 模型加载
        if not st.session_state.app_state["model_loaded"]:
            with st.spinner("加载LSTM模型与VMD..."):
                scaled_cases = st.session_state.app_state["processed_data"]["scaled_cases"]
                model_result = init_model_and_vmd(scaled_cases)
                if model_result["success"]:
                    st.session_state.app_state["model"] = model_result["model"]
                    st.session_state.app_state["vmd_transform"] = model_result["vmd_transform"]
                    st.session_state.app_state["model_loaded"] = True
                    st.success("✅ 模型与VMD初始化完成！")
                else:
                    st.error(f"❌ {model_result['msg']}")
        
        # 预测配置
        if st.session_state.app_state["model_loaded"]:
            st.subheader("3. 预测配置")
            last_monday = st.session_state.app_state["processed_data"]["last_monday"]
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "起始周（必须为周一）",
                    value=last_monday + timedelta(weeks=1),
                    min_value=last_monday + timedelta(weeks=1),
                    format="YYYY-MM-DD",
                    key="start_picker"
                )
            with col2:
                end_date = st.date_input(
                    "结束周（必须为周一）",
                    value=last_monday + timedelta(weeks=3),
                    min_value=start_date,
                    format="YYYY-MM-DD",
                    key="end_picker"
                )
            
            date_valid = True
            if start_date.weekday() != 0:
                st.warning("⚠️ 起始周必须选择周一！")
                date_valid = False
            if end_date.weekday() != 0:
                st.warning("⚠️ 结束周必须选择周一！")
                date_valid = False
            
            if date_valid and st.button("生成预测", key="predict_btn"):
                with st.spinner("正在预测..."):
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
                        st.error(f"❌ {pred_result['msg']}")
        
        # 预测结果
        if st.session_state.app_state["has_predictions"]:
            st.subheader("4. 预测结果")
            predictions = st.session_state.app_state["predictions"]
            
            # 修复：用单独的caption函数替代参数
            st.caption(f"共预测{len(predictions)}周")
            result_df = pd.DataFrame([
                {
                    "周范围": f"{p['start_date'].strftime('%Y-%m-%d')} ~ {p['end_date'].strftime('%Y-%m-%d')}",
                    "预测病例数": p["prediction"]
                } for p in predictions
            ])
            st.dataframe(result_df, use_container_width=True)
            
            total_cases = sum(p["prediction"] for p in predictions)
            st.markdown(f"### 📈 预测总病例数：**{total_cases} 例**")
            
            if st.button("重置预测结果", key="reset_btn"):
                st.session_state.app_state["has_predictions"] = False
                st.session_state.app_state["predictions"] = None
                st.success("已重置预测结果，可重新选择日期预测")


# 侧边栏
with st.sidebar:
    st.header("📋 使用说明")
    st.info("""
    1. 上传包含 **'Date'（日期）** 和 **'cases'（病例数）** 的CSV文件；
    2. 系统自动按“周一至周日”聚合周数据；
    3. 选择预测起始周/结束周（均需为周一）；
    4. 点击“生成预测”查看结果，支持重置重新预测。
    """)
    
    st.header("⚙️ 模型参数")
    st.write(f"VMD分解K值：{best_K}")
    st.write(f"滑动窗口大小：{window_size}")
    st.write(f"标准化方式：StandardScaler")
    st.write(f"LSTM模型路径：{MODEL_PATH}")


if __name__ == "__main__":
    main()
