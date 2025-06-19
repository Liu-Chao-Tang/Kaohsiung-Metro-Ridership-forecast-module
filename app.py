import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from forecast_core import load_data, train_model, forecast, auto_model, calculate_metrics, plot_acf_pacf, summarize_model_quality
from io import BytesIO

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="高雄捷運-運量預測系統", layout="wide")
st.markdown("<h1 style='text-align: center;'>高雄捷運-運量預測系統</h1>", unsafe_allow_html=True)

@st.cache_data
def cached_load():
    return load_data("data/daily_volume.xlsx")

df = cached_load()

all_numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
volume_cols = [col for col in all_numeric_columns if '運量' in col]
other_cols = [col for col in all_numeric_columns if col not in volume_cols]
sorted_columns = volume_cols + other_cols

default_target = '總運量' if '總運量' in sorted_columns else sorted_columns[0]

st.header("1. 預測項目選擇")
target_col = st.selectbox("請選擇要預測的欄位：", sorted_columns, index=sorted_columns.index(default_target))
exog_options = [col for col in df.columns if col != target_col]

last_actual_date = df[target_col].replace(0, np.nan).dropna().index.max()
min_date = df.index.min()
max_date = last_actual_date if pd.notnull(last_actual_date) else df.index.max()

st.header("2. 預測模型選擇")
mode = st.radio("", ("專家模式（自動選擇最優模型運算）", "自訂模式"))

model_type = "SARIMAX"
if mode == "自訂模式":
    st.subheader("自訂模式：請選擇模型類型")
    model_type = st.selectbox("選擇模型：", ["AR", "MA", "ARIMA", "SARIMAX"], index=3, format_func=lambda x: {
        "AR": "AR (自迴歸模型)",
        "MA": "MA (移動平均模型)",
        "ARIMA": "ARIMA (整合移動平均自迴歸模型)",
        "SARIMAX": "SARIMAX (季節性整合移動平均自迴歸模型)"
    }[x])

    if model_type == "AR":
        p = st.number_input("AR p", min_value=1, value=1, step=1)
        d = q = P = D = Q = S = 0
    elif model_type == "MA":
        q = st.number_input("MA q", min_value=1, value=1, step=1)
        p = d = P = D = Q = S = 0
    elif model_type == "ARIMA":
        p = st.number_input("AR p", min_value=0, value=1, step=1)
        d = st.number_input("差分 d", min_value=0, value=1, step=1)
        q = st.number_input("MA q", min_value=0, value=1, step=1)
        P = D = Q = S = 0
    else:
        p = st.number_input("AR p", min_value=0, value=1, step=1)
        d = st.number_input("差分 d", min_value=0, value=1, step=1)
        q = st.number_input("MA q", min_value=0, value=1, step=1)
        P = st.number_input("季節性 AR P", min_value=0, value=1, step=1)
        D = st.number_input("季節性差分 D", min_value=0, value=1, step=1)
        Q = st.number_input("季節性 MA Q", min_value=0, value=1, step=1)
        S = st.number_input("季節性週期 S (天)", min_value=1, value=7, step=1)
else:
    p = d = q = P = D = Q = S = None

st.header("3. 影響因子(外生變數)選擇")
use_exog = st.radio("", ("否(僅以運量歷史資料進行預測)", "是(考量其他影響因素)")) == "是(考量其他影響因素)"
selected_exog = []
if use_exog:
    st.markdown("請勾選要使用的外生變數：")
    for i, exog in enumerate(exog_options, start=1):
        label = f"變數{i}-{exog}"
        if st.checkbox(label, key=exog):
            selected_exog.append(exog)

st.header("4. 現有運量資料範圍")
st.markdown(f"資料日期範圍：**{min_date.date()}** 至 **{max_date.date()}**")

with st.form("forecast_form"):
    st.header("5. 模式訓練資料範圍設定")

    st.markdown("開始日期")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        start_year = st.selectbox("年", list(range(min_date.year, max_date.year + 1)), index=list(range(min_date.year, max_date.year + 1)).index(min_date.year), key="start_year")
    with col2:
        start_month = st.selectbox("月", list(range(1,13)), index=min_date.month-1, key="start_month")
    with col3:
        start_day = st.selectbox("日", list(range(1,32)), index=min_date.day-1, key="start_day")

    st.markdown("結束日期")
    col4, col5, col6 = st.columns([2, 1, 1])
    with col4:
        end_year = st.selectbox("年", list(range(min_date.year, max_date.year + 1)), index=list(range(min_date.year, max_date.year + 1)).index(max_date.year), key="end_year")
    with col5:
        end_month = st.selectbox("月", list(range(1,13)), index=max_date.month-1, key="end_month")
    with col6:
        end_day = st.selectbox("日", list(range(1,32)), index=max_date.day-1, key="end_day")

    # 組合日期
    try:
        train_start = pd.to_datetime(f"{start_year}-{start_month}-{start_day}")
    except:
        st.error("開始日期無效，請重新選擇")
        train_start = min_date

    try:
        train_end = pd.to_datetime(f"{end_year}-{end_month}-{end_day}")
    except:
        st.error("結束日期無效，請重新選擇")
        train_end = max_date

    n_forecast_days = st.slider("預測天數", min_value=1, max_value=365, value=7)

    submitted = st.form_submit_button("執行預測")

if submitted:
    try:
        st.info("開始執行預測...")
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        forecast_start_dt = train_end_dt + pd.Timedelta(days=1)
        forecast_end_dt = forecast_start_dt + pd.Timedelta(days=n_forecast_days - 1)

        if train_start_dt > train_end_dt:
            st.error("訓練開始日期不可晚於結束日期")
            st.stop()

        progress = st.progress(0, text="建立模型中...")

        if mode == "專家模式（自動選擇最優模型運算）":
            order, seasonal_order = auto_model(df, train_start_dt, train_end_dt, use_exog, selected_exog, target_col)
            st.success(f"自動模型參數：order={order}, seasonal_order={seasonal_order}")
            model_type = "SARIMAX"
        else:
            order = (p, d, q)
            seasonal_order = (P, D, Q, S)

        progress.progress(30, text="訓練模型中...")
        model_result = train_model(df, train_start_dt, train_end_dt, order, seasonal_order,
                                   use_exog, selected_exog, model_type, target_col)

        progress.progress(60, text="預測中...")
        df_forecast = forecast(model_result, df, forecast_start_dt, forecast_end_dt,
                               use_exog, selected_exog, model_type)

        actuals = df[target_col].reindex(df_forecast.index)
        df_result = df_forecast.copy()
        df_result['實際值'] = actuals
        df_result['預測誤差(%)'] = np.where(
            (df_result['實際值'].isna()) | (df_result['實際值'] == 0),
            np.nan,
            (abs(df_result['實際值'] - df_result['預測值']) / df_result['實際值']) * 100
        ).round(2)

        df_result = df_result[['實際值', '預測值', '下限', '上限', '預測誤差(%)']]
        mean_error = df_result['預測誤差(%)'].mean()

        st.subheader("預測結果")

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
        }).applymap(lambda v: 'color: green; font-weight: bold;' if pd.notna(v) else '', subset=['實際值']) \
          .applymap(color_error, subset=['預測誤差(%)'])

        st.dataframe(styled_df, use_container_width=True)
        st.markdown(f"<h4>📊 平均預測誤差(%)：{mean_error:.2f}%</h4>", unsafe_allow_html=True)

        if mean_error >= 10:
            st.warning("⚠️ 預測誤差偏高，建議檢查資料或調整模型")

        towrite = BytesIO()
        df_result.to_excel(towrite, index=True)
        st.download_button("📥 下載預測結果 Excel", towrite.getvalue(), file_name="forecast_result.xlsx")

        df_eval = df_result.dropna(subset=['實際值'])
        if not df_eval.empty:
            with st.expander("顯示模型績效指標總表"):
                metrics_df = calculate_metrics(model_result, df_eval['實際值'], df_eval['預測值'])
                st.dataframe(metrics_df.style.applymap(lambda v: 'color: red' if isinstance(v, str) and '❌' in v else ''))

                summary_text = summarize_model_quality(metrics_df)
                if "❌" in summary_text:
                    st.error(summary_text)
                else:
                    st.success(summary_text)

        with st.expander("📈 顯示運量預測圖表"):
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()

            ax1.plot(df_result.index, df_result['實際值'], label='實際值', color='blue', marker='o')
            ax1.plot(df_result.index, df_result['預測值'], label='預測值', color='orange', marker='s')
            ax1.fill_between(df_result.index, df_result['下限'], df_result['上限'], color='gray', alpha=0.2)

            ax2.bar(df_result.index, df_result['預測誤差(%)'], alpha=0.3, color='red', label='預測誤差(%)')
            ax2.set_ylim(0, 150)
            ax2.set_ylabel("預測誤差(%)")

            ax1.set_ylabel(target_col)
            ax1.grid(True)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax1.set_title("實際值、預測值及預測誤差(%)趨勢")

            for x, y in zip(df_result.index, df_result['實際值']):
                if pd.notna(y):
                    ax1.text(x, y, f"{int(y):,}", fontsize=8, color='blue', ha='left', va='bottom')
            for x, y in zip(df_result.index, df_result['預測值']):
                if pd.notna(y):
                    ax1.text(x, y, f"{int(y):,}", fontsize=8, color='orange', ha='right', va='bottom')
            for x, y in zip(df_result.index, df_result['預測誤差(%)']):
                if pd.notna(y):
                    ax2.text(x, y, f"{y:.1f}%", fontsize=7, color='red', ha='center', va='bottom')

            st.pyplot(fig)

        if not df_eval.empty:
            with st.expander("顯示模型 ACF / PACF 圖"):
                fig_acf_pacf = plot_acf_pacf(df, train_start_dt, train_end_dt, model_result)
                st.pyplot(fig_acf_pacf)

                
        with st.expander("📘 模型與指標名稱名詞解釋（英文 / 中文 / 定義說明）"):
            glossary_df = pd.DataFrame([
        ["AR,Autoregressive", "自迴歸模型", "Autoregressive Model: 利用自身過去的數據值來預測未來值。"],
        ["MA,Moving Average", "移動平均模型", "Moving Average Model: 利用過去誤差項的線性組合來預測。"],
        ["ARIMA,Autoregressive Integrated Moving Average", "整合移動平均自迴歸模型", "Autoregressive Integrated Moving Average Model: 包含差分處理以讓資料平穩的模型。"],
        ["SARIMAX,Seasonal Autoregressive Integrated Moving Average with eXogenous regressors", "季節性整合移動平均自迴歸模型", "Seasonal Autoregressive Integrated Moving Average with eXogenous regressors: 支援季節性與外生變數的 ARIMA 模型。"],
        ["Stationary R-squared", "平穩 R 平方", "在差分後資料上的模型解釋能力。越高越好。"],
        ["R-squared", "R 平方", "原始資料模型解釋能力。越高越好。"],
        ["RMSE,Root Mean Square Error", "均方根誤差", "衡量預測值與實際值平均誤差大小，單位與資料相同。越小越好。"],
        ["MAPE,Mean Absolute Percentage Error", "平均絕對百分比誤差", "百分比誤差衡量方式，越小越好。"],
        ["MAE,Mean Absolute Error", "平均絕對誤差", "平均預測絕對誤差。越小越好。"],
        ["Max AE,Maximum Absolute Error", "最大絕對誤差", "所有預測中最極端誤差。"],
        ["Normalized BIC,Bayesian Information Criterion", "正規化貝氏資訊準則", "懲罰模型複雜度後的擬合指標。越小越好。"],
        ["AIC,Akaike Information Criterion", "赤池資訊量準則", "衡量模型擬合度與複雜度的指標，數值越小模型越佳，避免過度擬合。"],
        ["Ljung-Box Q", "Ljung-Box Q 統計量", "檢驗殘差是否為白噪音。"],
        ["Degrees of Freedom", "自由度", "與 Q 統計檢定中使用的滯後階數有關。"],
        ["p-value", "顯著性", ">0.05 代表殘差無自相關（✅），≤0.05 有自相關（⚠️）"],
        ["p", "自我回歸階數 (AR)", "代表前幾期資料對當前值的解釋力。"],
        ["d", "差分階數 (I)", "資料差分次數，讓資料轉為平穩。"],
        ["q", "移動平均階數 (MA)", "使用前幾期誤差解釋目前值。"],
        ["P", "季節性 AR", "季節性資料中的 AR 階數。"],
        ["D", "季節性差分階數", "針對季節性趨勢所需的差分次數。"],
        ["Q", "季節性 MA", "季節性資料中的 MA 階數。"],
        ["S", "季節性週期", "季節性週期長度（如 7 日、12 月等）。"],
        ["ACF,Autocorrelation Function", "自相關函數", "度量目前值與過去值的線性關聯。"],
        ["PACF,Partial Autocorrelation Function", "偏自相關函數", "在控制其他滯後變數下，某滯後值的純粹貢獻。"]
            ], columns=["英文名稱", "中文名稱", "定義與說明"])
            st.dataframe(glossary_df, use_container_width=True)
            
        progress.progress(100, text="完成！")

    except Exception as e:
        st.exception(e)