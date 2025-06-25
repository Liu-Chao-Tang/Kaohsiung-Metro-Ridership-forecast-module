import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from forecast_core import ForecastModel
from io import BytesIO

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="高雄捷運-運量預測系統 ver. 1", layout="wide")
st.markdown("<h1 style='text-align: center;'>高雄捷運-運量預測系統 ver. 1</h1>", unsafe_allow_html=True)

@st.cache_data
def cached_load():
    df = pd.read_excel("data/daily_volume.xlsx", index_col=0, parse_dates=True)
    return df

df = cached_load()

all_numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
volume_cols = [col for col in all_numeric_columns if '運量' in col]

default_target = '總運量' if '總運量' in volume_cols else volume_cols[0]

st.header("1. 預測項目選擇")
target_col = st.selectbox(
    "預測項目（僅限運量相關欄位）：",
    volume_cols,
    index=volume_cols.index(default_target) if default_target in volume_cols else 0)
exog_options = [col for col in df.columns if col != target_col]

last_actual_date = df[target_col].replace(0, np.nan).dropna().index.max()
min_date = df.index.min()
max_date = last_actual_date if pd.notnull(last_actual_date) else df.index.max()

st.header("2. 預測模型選擇")
mode = st.radio("", ("專家模式（自動選擇最優模型運算）", "自訂模式"))

if mode == "專家模式（自動選擇最優模型運算）":
    m = 7
    search_speed = st.radio("模型搜尋模式", ["快速（運算時間短）", "精準（準確率較高）"])
    stepwise_mode = True if search_speed == "快速（運算時間短）" else False
else:
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
        m = S

st.header("3. 影響因子選擇(外生變數)")
use_exog = st.radio("", ("否(僅以歷史運量資料進行預測)", "是(考量其他影響因素)")) == "是(考量其他影響因素)"
selected_exog = []
if use_exog:
    st.markdown("請勾選要使用的外生變數：")
    for i, exog in enumerate(exog_options, start=1):
        label = f"變數{i}-{exog}"
        default_checked = any(k in exog for k in ["溫度", "降雨", "假日"])
        if st.checkbox(label, key=exog, value=default_checked):
            selected_exog.append(exog)

st.header("4. 現有運量資料範圍")
st.markdown(f"資料日期範圍：**{min_date.date()}** 至 **{max_date.date()}**")

with st.form("forecast_form"):
    st.header("5. 模式訓練與預測期間設定")
    train_start = st.date_input("訓練開始日", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    train_end = st.date_input("訓練結束日", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
    n_forecast_days = st.slider("預測天數", min_value=1, max_value=365, value=7)
    submitted = st.form_submit_button("執行預測")

if submitted:
    try:
        st.info("開始執行預測...")
    
        import time
        start_time = time.time()

        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        forecast_start_dt = train_end_dt + pd.Timedelta(days=1)
        forecast_end_dt = forecast_start_dt + pd.Timedelta(days=n_forecast_days - 1)

        if train_start_dt > train_end_dt:
            st.error("訓練開始日期不可晚於結束日期")
            st.stop()

        progress = st.progress(0, text="建立模型中...")

        fm = ForecastModel(df, target_col, use_exog, selected_exog)

        if mode == "專家模式（自動選擇最優模型運算）":
            import time
            t0 = time.time()
            order, seasonal_order = fm.auto_fit(train_start_dt, train_end_dt, m=m, expert_mode=True, stepwise_mode=stepwise_mode)
            t1 = time.time()
            st.write(f"auto_fit() 耗時: {t1 - t0:.2f} 秒")
            model_type = "SARIMAX"
            st.success(f"自動模型參數：order={order}, seasonal_order={seasonal_order}")
        else:
            order = (p, d, q)
            seasonal_order = (P, D, Q, S)

        t1 = time.time()
        elapsed1 = t1 - start_time
        est_total = elapsed1 / 0.3
        remaining = est_total - elapsed1
        progress.progress(30, text=f"訓練模型中... ⏳ 預估剩餘 {remaining:.1f} 秒")

        try:
            model_result = fm.fit(train_start_dt, train_end_dt, order, seasonal_order, model_type)
        except Exception as e:
            st.error("❌ 模型訓練失敗，請檢查參數或資料格式")
            st.stop()

        t2 = time.time()
        elapsed2 = t2 - start_time
        est_total = elapsed2 / 0.6
        remaining = est_total - elapsed2
        progress.progress(60, text=f"預測中... ⏳ 預估剩餘 {remaining:.1f} 秒")

        df_forecast = fm.forecast(forecast_start_dt, forecast_end_dt)
        
        progress.progress(100, text="✅ 預測完成")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.success(f"⏱️ 預測流程共花費時間：{elapsed_time:.2f} 秒")

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
        st.divider()
        st.download_button("📥 下載預測結果 Excel", towrite.getvalue(), file_name="forecast_result.xlsx")

        df_eval = df_result.dropna(subset=['實際值'])
        if not df_eval.empty:
            st.divider()
            with st.expander("顯示模型績效指標總表"):
                metrics = fm.calculate_metrics(df_eval['實際值'], df_eval['預測值'])

                metrics_info = {
                    'MAPE (%)': {
                        '標準': '<=10%',
                        '說明': '平均絕對百分比誤差，衡量預測誤差相對於實際值的百分比。',
                        '判斷': lambda v: '好' if v <= 10 else ('普通' if v <= 20 else '差')
                    },
                    'R-squared': {
                        '標準': '>=0.8',
                        '說明': '模型解釋變異的比例，越高代表模型越好。',
                        '判斷': lambda v: '好' if v >= 0.8 else ('普通' if v >= 0.6 else '差')
                    },
                    'Adjusted R-squared': {
                        '標準': '>=0.8',
                        '說明': '調整參數數量後的R平方，避免過度擬合。',
                        '判斷': lambda v: '好' if v >= 0.8 else ('普通' if v >= 0.6 else '差')
                    },
                    'Stabilized R-squared': {
                        '標準': '>=0.8',
                        '說明': '在差分後資料上的模型解釋能力。',
                        '判斷': lambda v: '好' if v >= 0.8 else ('普通' if v >= 0.6 else '差')
                    },
                    'RMSE': {
                        '標準': '越低越好',
                        '說明': '均方根誤差，反映預測誤差大小。',
                        '判斷': lambda v: '好' if v < 10000 else ('普通' if v < 20000 else '差')
                    },
                    'MAE': {
                        '標準': '越低越好',
                        '說明': '平均誤差值，越低越好。',
                        '判斷': lambda v: '好' if v < 10000 else ('普通' if v < 20000 else '差')
                    },
                    'Max AE': {
                        '標準': '越低越好',
                        '說明': '最大誤差值，偏差最大情況。',
                        '判斷': lambda v: '好' if v < 20000 else ('普通' if v < 40000 else '差')
                    },
                    'Normalized BIC': {
                        '標準': '越低越好',
                        '說明': '每筆樣本的BIC指標，越低越好。',
                        '判斷': lambda v: '好' if v < 50 else ('普通' if v < 100 else '差')
                    },
                    'AIC': {
                        '標準': '越低越好',
                        '說明': '衡量模型適配度與簡單性的資訊量準則。',
                        '判斷': lambda v: '好' if v < 10000 else ('普通' if v < 20000 else '差')
                    },
                    'Ljung-Box p-value': {
                        '標準': '>0.05為佳',
                        '說明': '檢驗殘差是否為白噪音。',
                        '判斷': lambda v: '好' if v > 0.05 else '差'
                    },
                    'Durbin-Watson': {
                        '標準': '約 2 為佳',
                        '說明': 'Durbin-Watson 統計量用於檢驗殘差自相關，約等於2表示無自相關。',
                        '判斷': lambda v: '好' if 1.5 <= v <= 2.5 else '差'
                    },
                    '樣本數 N': {
                        '標準': '越大越穩定',
                        '說明': '評估樣本數，用於衡量訓練資料量。',
                        '判斷': lambda v: '好' if v >= 30 else ('普通' if v >= 10 else '差')
                    }
                }

                rows = []
                priority_order = ['MAPE (%)', 'R-squared', 'Adjusted R-squared', 'Stabilized R-squared',
                                  'RMSE', 'MAE', 'Max AE', 'Normalized BIC', 'AIC', 'Ljung-Box p-value', 'Durbin-Watson', '樣本數 N']

                for key in priority_order:
                    val = metrics.get(key, np.nan)
                    std = metrics_info[key]['標準']
                    desc = metrics_info[key]['說明']

                    if isinstance(val, (int, float, np.float64, np.float32)) and not np.isnan(val):
                        judge = metrics_info[key]['判斷'](val)
                        val_str = f"{val:.4f}" if "p-value" in key or "%" in key else f"{val:.2f}"
                    else:
                        judge = '-'
                        val_str = str(val)

                    rows.append([key, val_str, std, judge, desc])

                df_metrics = pd.DataFrame(rows, columns=['指標', '模型實際值', '判斷標準', '判別結果', '指標說明'])

                st.dataframe(
                    df_metrics.style.applymap(
                        lambda v: 'color: red' if v == '差' else ('color: orange' if v == '普通' else 'color: green'),
                        subset=['判別結果']
                    ),
                    use_container_width=True
                )

                summary_text = fm.summarize_quality(metrics)
                if "❌" in summary_text:
                    st.error(summary_text)
                else:
                    st.success(summary_text)

        # ----- 只在展開區塊內顯示預測圖 -----
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

        # ----- 合併 ACF/PACF 原始與殘差圖 -----
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        with st.expander("📊 顯示 ACF / PACF 分析圖（原始資料與殘差）"):
            orig_series = df[target_col].loc[train_start_dt:train_end_dt].dropna()
            resid_series = None
            if hasattr(model_result, "resid"):
                resid_series = model_result.resid.dropna()

            fig_acf_pacf, axes = plt.subplots(2, 2, figsize=(14, 8))

            # 原始資料 ACF
            plot_acf(orig_series, ax=axes[0,0], lags=40)
            axes[0,0].set_title("原始資料 ACF")

            # 原始資料 PACF
            plot_pacf(orig_series, ax=axes[0,1], lags=40, method='ywm')
            axes[0,1].set_title("原始資料 PACF")

            # 殘差 ACF
            if resid_series is not None and len(resid_series) > 0:
                plot_acf(resid_series, ax=axes[1,0], lags=40)
                axes[1,0].set_title("殘差 ACF")
            else:
                axes[1,0].axis('off')

            # 殘差 PACF
            if resid_series is not None and len(resid_series) > 0:
                plot_pacf(resid_series, ax=axes[1,1], lags=40, method='ywm')
                axes[1,1].set_title("殘差 PACF")
            else:
                axes[1,1].axis('off')

            plt.tight_layout()
            st.pyplot(fig_acf_pacf)
        if not df_eval.empty:
            with st.expander("📊 模型統計摘要"):
                st.code(str(model_result.summary()), language='text')

        with st.expander("📘 附錄-模型名詞解釋（英文 / 中文 / 定義說明）"):
            glossary_df = pd.DataFrame([
                ["AR,Autoregressive", "自迴歸模型", "Autoregressive Model: 利用自身過去的數據值來預測未來值。"],
                ["MA,Moving Average", "移動平均模型", "Moving Average Model: 利用過去誤差項的線性組合來預測。"],
                ["ARIMA,Autoregressive Integrated Moving Average", "整合移動平均自迴歸模型", "Autoregressive Integrated Moving Average Model: 包含差分處理以讓資料平穩。"],
                ["SARIMAX,Seasonal ARIMA with Exogenous Variables", "季節性ARIMA外生變數模型", "在ARIMA基礎上加上季節性成分與外生變數。"],
                ["MAPE", "平均絕對百分比誤差", "衡量預測值與實際值之間的平均百分比誤差。"],
                ["R-squared", "決定係數", "衡量模型解釋變異程度的指標，介於0到1之間。"],
                ["RMSE", "均方根誤差", "誤差平方的平均後再開根號，表示平均預測誤差。"]
            ], columns=["英文縮寫", "中文名稱", "定義說明"])

            st.dataframe(glossary_df, use_container_width=True)

    except Exception as e:
        st.error(f"執行預測發生錯誤: {e}")
