import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from forecast_core import load_data, train_model, forecast, auto_model, calculate_metrics, plot_acf_pacf

# ✅ 加入中文字型設定（適用於 Windows）
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="高雄捷運-運量預測系統", layout="wide")  # 修改標題
st.title("高雄捷運-運量預測系統")  # 修改標題

# 載入資料
data_path = "data/daily_volume.xlsx"
df = load_data(data_path)

exog_options = ['變數1-特殊節日', '變數2-大型活動', '變數3-颱風假']

last_actual_date = df['總運量'].replace(0, np.nan).dropna().index.max()
min_date = df.index.min()
max_date = last_actual_date if pd.notnull(last_actual_date) else df.index.max()

# 1. 運量預測模式選擇
st.header("1. 運量預測模式選擇")
mode = st.radio(
    "",
    ("專家模式（自動選擇最優模型運算）", "自訂模式")
)

# 自訂模式參數輸入
if mode == "自訂模式":
    st.subheader("自訂模式：請設定模型參數")
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("AR p", min_value=0, value=1, step=1)
        P = st.number_input("季節性 AR P", min_value=0, value=1, step=1)
    with col2:
        d = st.number_input("差分 d", min_value=0, value=1, step=1)
        D = st.number_input("季節性差分 D", min_value=0, value=1, step=1)
    with col3:
        q = st.number_input("MA q", min_value=0, value=1, step=1)
        Q = st.number_input("季節性 MA Q", min_value=0, value=1, step=1)
    S = st.number_input("季節性週期 S (天)", min_value=1, value=7, step=1)
else:
    p = d = q = P = D = Q = S = None

# 2. 是否使用外生變數
st.header("2. 是否使用外生變數")
use_exog = st.radio("", ("否", "是")) == "是"

selected_exog = []
if use_exog:
    st.markdown("請勾選要使用的外生變數：")
    for exog in exog_options:
        if st.checkbox(exog, key=exog):
            selected_exog.append(exog)

# 3. 歷史資料範圍
st.header("3. 歷史資料範圍")
st.markdown(f"資料日期範圍：**{min_date.date()}** 至 **{max_date.date()}**")

with st.form("forecast_form"):
    # 4. 預測模型建模資料區間
    st.header("4. 預測模型建模資料區間")
    train_start = st.date_input("開始日期", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    train_end = st.date_input("結束日期", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

    # 起始預測日期 = 訓練結束日 + 1 天
    forecast_start = train_end + pd.Timedelta(days=1)

    # 5. 運量預測天數（最大90天）
    st.header("5. 運量預測天數")
    n_forecast_days = st.number_input("請輸入預測天數", min_value=1, max_value=90, value=7, step=1)

    submitted = st.form_submit_button("執行預測")

if submitted:
    try:
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)

        # 預測起訖日期
        forecast_start_dt = train_end_dt + pd.Timedelta(days=1)
        forecast_end_dt = forecast_start_dt + pd.Timedelta(days=n_forecast_days - 1)

        # 檢查日期合理性
        if train_start_dt > train_end_dt:
            st.error("訓練開始日期不可晚於結束日期")
            st.stop()

        if mode == "自訂模式":
            order = (p, d, q)
            seasonal_order = (P, D, Q, S)
        else:
            order, seasonal_order = auto_model(df, train_start_dt, train_end_dt, use_exog, selected_exog)
            if order is None:
                st.error("無法自動選擇模型，請改用自訂模式設定參數。")
                st.stop()
            st.success(f"自動選擇模型參數：order={order}, seasonal_order={seasonal_order}")

        model_result = train_model(df, train_start_dt, train_end_dt, order, seasonal_order, use_exog, selected_exog)
        df_forecast = forecast(model_result, df, forecast_start_dt, forecast_end_dt, use_exog, selected_exog)

        actuals = df['總運量'].reindex(df_forecast.index)
        df_result = df_forecast.copy()
        df_result['實際值'] = actuals

        df_result['預測誤差(%)'] = np.where(
            (df_result['實際值'].isna()) | (df_result['實際值'] == 0),
            np.nan,
            (abs(df_result['實際值'] - df_result['預測值']) / df_result['實際值']) * 100
        ).round(2)

        mean_error = df_result['預測誤差(%)'].mean()

        df_result = df_result[['實際值', '預測值', '下限', '上限', '預測誤差(%)']]

        st.subheader("預測結果")

        def color_actual(val):
            return 'color: blue' if pd.notnull(val) else ''

        def color_error(val):
            if pd.isna(val):
                return ''
            return 'color: red' if val >= 10 else ''

        styled_df = df_result.style.format({
            '實際值': '{:,.0f}',
            '預測值': '{:,.0f}',
            '下限': '{:,.0f}',
            '上限': '{:,.0f}',
            '預測誤差(%)': '{:.2f}%'
        }).applymap(color_actual, subset=['實際值']).applymap(color_error, subset=['預測誤差(%)'])

        st.dataframe(styled_df)

        st.markdown(f"**平均預測誤差(%)：** {mean_error:.2f}%")

        df_eval = df_result.dropna(subset=['實際值'])
        if not df_eval.empty:
            metrics = calculate_metrics(model_result, df_eval['實際值'], df_eval['預測值'])
            st.subheader("績效指標")
            st.table(metrics)
        else:
            st.info("預測區間無實際值，無法計算績效指標。")

        # 畫圖：折線圖 + 柱狀圖 + 數值標籤 + 副軸範圍固定
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(df_result.index, df_result['實際值'], label='實際值', color='blue', marker='o')
        ax1.plot(df_result.index, df_result['預測值'], label='預測值', color='orange', marker='s')
        ax1.fill_between(df_result.index, df_result['下限'], df_result['上限'], color='gray', alpha=0.2)
        ax1.set_ylabel("總運量")
        ax1.grid(True)

        for x, y in zip(df_result.index, df_result['實際值']):
            if pd.notna(y):
                ax1.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=8, color='blue')

        for x, y in zip(df_result.index, df_result['預測值']):
            if pd.notna(y):
                ax1.text(x, y, f'{y:,.0f}', ha='center', va='top', fontsize=8, color='orange')

        ax2 = ax1.twinx()
        bars = ax2.bar(df_result.index, df_result['預測誤差(%)'], color='red', alpha=0.3, label='預測誤差(%)')
        ax2.set_ylabel("預測誤差 (%)")
        ax2.set_ylim(0, 150)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    color='red',
                    rotation=0
                )

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        ax1.set_title("實際值、預測值趨勢及預測誤差(%)")
        st.pyplot(fig)

        # ACF 與 PACF 圖
        st.subheader("ACF 與 PACF 圖")
        fig_acf_pacf = plot_acf_pacf(df, train_start_dt, train_end_dt, model_result)
        st.pyplot(fig_acf_pacf)

    except Exception as e:
        st.error(f"執行錯誤：{e}")
