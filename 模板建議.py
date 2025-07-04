import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="高雄捷運運量預測", layout="wide")

# 左側欄：操作步驟區（上下滾動）
with st.sidebar:
    st.title("操作步驟")
    mode = st.radio("1️⃣ 選擇預測模式", options=["自動模式", "專家模式", "自訂模式"])
    
    st.markdown("---")
    
    # 範例參數設定
    n_forecast = st.slider("2️⃣ 選擇預測天數", min_value=7, max_value=60, value=30, step=1)
    
    st.markdown("---")
    
    # 啟動按鈕
    start = st.button("▶️ 開始模型訓練與預測")

# 右側主欄：預測結果、指標、圖表
col1, col2 = st.columns([1, 3])  # 1:3 比例分配欄寬

with col1:
    st.markdown("<h2 style='color:#FF6B00;'>績效指標</h2>", unsafe_allow_html=True)
    # 假設績效指標
    r2 = 0.87
    mape = 12.5
    rmse = 150.3
    
    # 強調色彩與大字體
    st.markdown(f"<h1 style='color:#1F77B4;'>{r2:.2f}</h1><p>R²</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:#FF7F0E;'>{mape:.1f}%</h1><p>MAPE</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:#2CA02C;'>{rmse:.1f}</h1><p>RMSE</p>", unsafe_allow_html=True)

    # 模型日誌收合區
    with st.expander("📜 模型執行日誌", expanded=False):
        log_df = pd.DataFrame({
            "時間": ["2025-07-04 16:00", "2025-07-04 16:30"],
            "模式": ["自動", "專家"],
            "MAPE(%)": [12.5, 15.2],
            "R2": [0.87, 0.82]
        })
        st.dataframe(log_df)
        csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ 下載日誌 CSV", csv, "model_log.csv", "text/csv")

    # PDF報告下載
    st.markdown("### 📄 報告下載")
    dummy_pdf = b"%PDF-1.4...pdf bytes here..."
    st.download_button(
        label="⬇️ 下載 PDF 報告",
        data=dummy_pdf,
        file_name="model_report.pdf",
        mime="application/pdf"
    )

with col2:
    st.markdown("<h2 style='color:#FF6B00;'>🚊 未來運量預測</h2>", unsafe_allow_html=True)
    
    if start:
        with st.spinner("模型訓練與預測中..."):
            time.sleep(2)  # 模擬運算
        st.balloons()
        st.success("預測完成！")

        # 模擬預測資料
        dates = pd.date_range(start=pd.Timestamp.today(), periods=n_forecast)
        pred_values = np.random.randint(3000, 6000, size=n_forecast)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, pred_values, marker='o', color="#FF6B00")
        ax.set_title(f"未來 {n_forecast} 天捷運運量預測")
        ax.set_ylabel("運量")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("請先在左側欄點擊「開始模型訓練與預測」")

