import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima

def load_data(filepath):
    df = pd.read_excel(filepath, index_col=0, parse_dates=True)
    return df

def train_model(df, train_start, train_end, order, seasonal_order, use_exog, selected_exog):
    train_df = df.loc[train_start:train_end]
    endog = train_df['總運量']
    exog = train_df[selected_exog] if use_exog and selected_exog else None

    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def forecast(model_fit, df, forecast_start, forecast_end, use_exog, selected_exog):
    forecast_index = pd.date_range(start=forecast_start, end=forecast_end)
    if use_exog and selected_exog:
        exog_forecast = df.loc[forecast_index, selected_exog]
    else:
        exog_forecast = None

    pred = model_fit.get_forecast(steps=len(forecast_index), exog=exog_forecast)
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int()

    df_forecast = pd.DataFrame({
        '預測值': pred_mean,
        '下限': conf_int.iloc[:, 0],
        '上限': conf_int.iloc[:, 1]
    }, index=forecast_index)
    return df_forecast

def auto_model(df, train_start, train_end, use_exog, selected_exog):
    train_df = df.loc[train_start:train_end]
    endog = train_df['總運量']
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

    order = stepwise_model.order
    seasonal_order = stepwise_model.seasonal_order

    return order, seasonal_order

def calculate_metrics(model_fit, actual, predicted):
    residuals = actual - predicted
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs(residuals / actual)) * 100
    max_ae = np.max(np.abs(residuals))
    r2 = r2_score(actual, predicted)
    n = len(actual)
    p = len(model_fit.params)

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan

    bic = model_fit.bic if hasattr(model_fit, 'bic') else np.nan
    aic = model_fit.aic if hasattr(model_fit, 'aic') else np.nan
    normalized_bic = bic / n if not np.isnan(bic) else np.nan

    try:
        diff_actual = actual.diff().dropna()
        diff_pred = predicted.diff().dropna()
        stationary_r2 = r2_score(diff_actual, diff_pred)
    except:
        stationary_r2 = np.nan

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)
        ljung_box_q = lb_test['lb_stat'].values[0]
        ljung_box_p = lb_test['lb_pvalue'].values[0]
    except:
        ljung_box_q = np.nan
        ljung_box_p = np.nan

    metrics = {
        'Stationary R-squared': f"{stationary_r2:.4f}" if not np.isnan(stationary_r2) else 'N/A',
        'R-squared': f"{r2:.4f}",
        'Adjusted R-squared': f"{adj_r2:.4f}" if not np.isnan(adj_r2) else 'N/A',
        'RMSE': f"{rmse:.2f}",
        'MAPE (%)': f"{mape:.2f}",
        'MAE': f"{mae:.2f}",
        'Max AE': f"{max_ae:.2f}",
        'Normalized BIC': f"{normalized_bic:.4f}" if not np.isnan(normalized_bic) else 'N/A',
        'AIC': f"{aic:.4f}" if not np.isnan(aic) else 'N/A',
        'Ljung-Box Q': f"{ljung_box_q:.4f}" if not np.isnan(ljung_box_q) else 'N/A',
        'Ljung-Box p-value': f"{ljung_box_p:.4f}" if not np.isnan(ljung_box_p) else 'N/A'
    }
    return metrics

def plot_acf_pacf(df, train_start, train_end, model_fit=None):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    train_df = df.loc[train_start:train_end]
    series = train_df['總運量']

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
