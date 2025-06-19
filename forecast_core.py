import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima

def load_data(filepath):
    df = pd.read_excel(filepath, index_col=0, parse_dates=True)
    return df

def train_model(df, train_start, train_end, order, seasonal_order, use_exog, selected_exog, model_type, target):
    train_df = df.loc[train_start:train_end]
    endog = train_df[target]
    exog = train_df[selected_exog] if use_exog and selected_exog else None

    if model_type == "AR":
        model = AutoReg(endog, lags=order[0], old_names=False)
        model_fit = model.fit()
    elif model_type == "MA":
        model = ARIMA(endog, order=(0, 0, order[2]))
        model_fit = model.fit()
    elif model_type == "ARIMA":
        model = ARIMA(endog, order=order)
        model_fit = model.fit()
    else:
        model = SARIMAX(endog, order=order, seasonal_order=seasonal_order, exog=exog,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)

    return model_fit

def forecast(model_fit, df, forecast_start, forecast_end, use_exog, selected_exog, model_type):
    forecast_index = pd.date_range(start=forecast_start, end=forecast_end)

    if model_type in ["AR", "MA", "ARIMA"]:
        pred = model_fit.get_prediction(start=forecast_index[0], end=forecast_index[-1])
    else:
        exog_forecast = df.loc[forecast_index, selected_exog] if use_exog and selected_exog else None
        pred = model_fit.get_forecast(steps=len(forecast_index), exog=exog_forecast)

    pred_mean = pred.predicted_mean
    try:
        conf_int = pred.conf_int()
    except:
        conf_int = pd.DataFrame({"lower": pred_mean * 0.9, "upper": pred_mean * 1.1}, index=forecast_index)

    df_forecast = pd.DataFrame({
        '預測值': pred_mean,
        '下限': conf_int.iloc[:, 0],
        '上限': conf_int.iloc[:, 1]
    }, index=forecast_index)
    return df_forecast

def auto_model(df, train_start, train_end, use_exog, selected_exog, target):
    train_df = df.loc[train_start:train_end]
    endog = train_df[target]
    exog = train_df[selected_exog] if use_exog and selected_exog else None

    stepwise_model = auto_arima(
        endog,
        exogenous=exog,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=None,
        seasonal=True,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=None,
        m=7,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_jobs=1
    )

    return stepwise_model.order, stepwise_model.seasonal_order

def summarize_model_quality(metrics_df):
    flags = metrics_df['結果'].astype(str).values
    fail_flags = [f for f in flags if '❌' in f or '⚠' in f]
    if not fail_flags:
        return "✅ 模型表現良好，可直接應用於實務預測。"

    suggestions = []
    if '❌' in metrics_df.loc['R-squared', '結果']:
        suggestions.append("R平方過低，建議增加訓練資料量或加入其他外生變數。")
    if '❌' in metrics_df.loc['Adjusted R-squared', '結果']:
        suggestions.append("調整後R平方過低，模型可能過於簡單或資料不足。")
    if '❌' in metrics_df.loc['MAPE (%)', '結果']:
        suggestions.append("MAPE過高，建議考慮目標變數轉換或加入更多外部影響因素。")
    if '❌' in metrics_df.loc['Ljung-Box p-value', '結果']:
        suggestions.append("殘差自相關明顯，建議提高模型的AR或季節性階數。")

    if '⚠' in flags:
        suggestions.append("部分指標為普通，模型表現尚可，但仍建議密切監控預測誤差。")

    summary = "❌ 模型存在以下問題：\n" + "\n".join(f"- {s}" for s in suggestions)
    return summary

def calculate_metrics(model_fit, actual, predicted):
    residuals = actual - predicted
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs(residuals / actual)) * 100
    max_ae = np.max(np.abs(residuals))
    r2 = r2_score(actual, predicted)
    n = len(actual)
    p = len(model_fit.params) if hasattr(model_fit, 'params') else 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    stabilized_r2 = 1 - (np.var(residuals) / np.var(actual)) if np.var(actual) != 0 else np.nan

    bic = getattr(model_fit, 'bic', np.nan)
    aic = getattr(model_fit, 'aic', np.nan)
    normalized_bic = bic / n if not np.isnan(bic) else np.nan

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)
        ljung_box_q = lb_test['lb_stat'].values[0]
        ljung_box_p = lb_test['lb_pvalue'].values[0]
    except:
        ljung_box_q = np.nan
        ljung_box_p = np.nan

    metric_desc = {
        'R-squared': '判斷模型解釋變異能力，越接近1越好',
        'Adjusted R-squared': '調整後的R方，考慮變數數量，越高越好',
        'Stabilized R-squared': '平穩化R²衡量殘差穩定程度，越高越佳',
        'MAPE (%)': '平均絕對百分比誤差，數值越低越佳',
        'RMSE': '均方根誤差，越低越好',
        'MAE': '平均絕對誤差，越低越好',
        'Max AE': '最大絕對誤差，越低越好',
        'Normalized BIC': '貝氏資訊準則，數值越低模型越佳',
        'AIC': '赤池資訊準則，數值越低模型越佳',
        'Ljung-Box Q': '殘差自相關統計量',
        'Ljung-Box p-value': '殘差自相關檢定p值'
    }

    metrics = {
        'R-squared': f"{r2:.4f}",
        'Adjusted R-squared': f"{adj_r2:.4f}" if not np.isnan(adj_r2) else 'N/A',
        'Stabilized R-squared': f"{stabilized_r2:.4f}" if not np.isnan(stabilized_r2) else 'N/A',
        'MAPE (%)': f"{mape:.2f}",
        'RMSE': f"{rmse:.2f}",
        'MAE': f"{mae:.2f}",
        'Max AE': f"{max_ae:.2f}",
        'Normalized BIC': f"{normalized_bic:.4f}" if not np.isnan(normalized_bic) else 'N/A',
        'AIC': f"{aic:.4f}" if not np.isnan(aic) else 'N/A',
        'Ljung-Box Q': f"{ljung_box_q:.4f}" if not np.isnan(ljung_box_q) else 'N/A',
        'Ljung-Box p-value': f"{ljung_box_p:.4f}" if not np.isnan(ljung_box_p) else 'N/A'
    }

    standards_text = {
        'MAPE (%)': "<5%：佳，<10%：可接受，<20%：普通，>20%：差",
        'R-squared': "建議 ≥0.5",
        'Adjusted R-squared': "建議 ≥0.5",
        'Stabilized R-squared': "建議 ≥0.5",
        'Ljung-Box p-value': "建議 >0.05，殘差無自相關"
    }

    annotated_metrics = {}
    for k, v in metrics.items():
        std = standards_text.get(k, "-")
        flag = ''
        try:
            val_float = float(v.replace('%', '')) if '%' in v else float(v)
            if k == 'MAPE (%)':
                if val_float < 5:
                    flag = "✅佳"
                elif val_float < 10:
                    flag = "✅可接受"
                elif val_float < 20:
                    flag = "⚠普通"
                else:
                    flag = "❌差"
            elif k in ['R-squared', 'Adjusted R-squared', 'Stabilized R-squared']:
                flag = "✅" if val_float >= 0.5 else "❌"
            elif k == 'Ljung-Box p-value':
                flag = "✅" if val_float > 0.05 else "❌"
        except:
            flag = "N/A"

        annotated_metrics[k] = {
            "值": v,
            "建議標準": std,
            "結果": flag,
            "說明": metric_desc.get(k, "")
        }

    return pd.DataFrame(annotated_metrics).T

def plot_acf_pacf(df, train_start, train_end, model_fit=None):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    train_df = df.loc[train_start:train_end]
    series = train_df.select_dtypes(include=[np.number]).iloc[:, 0]
    plot_acf(series.dropna(), ax=axes[0, 0], lags=30)
    axes[0, 0].set_title('原始資料 ACF')
    plot_pacf(series.dropna(), ax=axes[0, 1], lags=30)
    axes[0, 1].set_title('原始資料 PACF')
    if model_fit:
        resid = model_fit.resid.dropna()
        plot_acf(resid, ax=axes[1, 0], lags=30)
        axes[1, 0].set_title('模型殘差 ACF')
        plot_pacf(resid, ax=axes[1, 1], lags=30)
        axes[1, 1].set_title('模型殘差 PACF')
    plt.tight_layout()
    return fig


