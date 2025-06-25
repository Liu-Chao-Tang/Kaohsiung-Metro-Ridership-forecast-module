import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ForecastModel:
    def __init__(self, df, target_col, use_exog=False, exog_cols=None):
        self.df = df
        self.target_col = target_col
        self.use_exog = use_exog
        self.exog_cols = exog_cols if exog_cols else []
        self.model_fit = None
        self.order = None
        self.seasonal_order = None
        self.model_type = None
        self.train_start_date = None
        self.train_end_date = None

    def load_data(self, filepath):
        self.df = pd.read_excel(filepath, index_col=0, parse_dates=True)

    def auto_fit(self, train_start, train_end, m=7, expert_mode=False, stepwise_mode=True):
        train_df = self.df.loc[train_start:train_end]
        endog = train_df[self.target_col]
        exog = train_df[self.exog_cols] if self.use_exog and self.exog_cols else None

        seasonal = expert_mode or (m != 0)

        max_p = 10
        max_q = 10
        max_P = 2
        max_Q = 2

        stepwise_model = auto_arima(
            endog, exogenous=exog, seasonal=seasonal, m=m,
            start_p=0, max_p=max_p, start_q=0, max_q=max_q,
            start_P=0, max_P=max_P, start_Q=0, max_Q=max_Q,
            d=None, D=None,          # 讓auto_arima判斷差分階數
            max_d=2, max_D=1,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=stepwise_mode,
            n_jobs=1
        )
        self.order = stepwise_model.order
        self.seasonal_order = stepwise_model.seasonal_order
        return self.order, self.seasonal_order

    def fit(self, train_start, train_end, order, seasonal_order, model_type="SARIMAX"):
        self.train_start_date = train_start
        self.train_end_date = train_end

        train_df = self.df.loc[train_start:train_end]
        endog = train_df[self.target_col]
        exog = train_df[self.exog_cols] if self.use_exog and self.exog_cols else None

        self.order = order
        self.seasonal_order = seasonal_order
        self.model_type = model_type

        if model_type == "AR":
            model = AutoReg(endog, lags=order[0], old_names=False)
            self.model_fit = model.fit()
        elif model_type == "MA":
            model = ARIMA(endog, order=(0, 0, order[2]))
            self.model_fit = model.fit()
        elif model_type == "ARIMA":
            model = ARIMA(endog, order=order)
            self.model_fit = model.fit()
        else:
            model = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                            exog=exog, enforce_stationarity=False, enforce_invertibility=False)
            self.model_fit = model.fit(disp=False)
        return self.model_fit

    def forecast(self, forecast_start, forecast_end):
        if self.model_fit is None:
            raise ValueError("\u6a21\u578b\u5c1a\u672a\u8a13\u7df4\u5b8c\u6210")

        full_index = pd.date_range(forecast_start, forecast_end)
        steps = len(full_index)

        exog_forecast = None
        if self.use_exog and self.exog_cols:
            exog_forecast = self.df[self.exog_cols].reindex(full_index)

        if self.model_type == "AR":
            start = len(self.model_fit.model.endog)
            end = start + steps - 1
            pred_mean = self.model_fit.predict(start=start, end=end)
            pred_mean.index = full_index
            resid_std = np.std(self.model_fit.resid)
            lower = pred_mean - 1.96 * resid_std
            upper = pred_mean + 1.96 * resid_std

        elif self.model_type in ["MA", "ARIMA"]:
            pred = self.model_fit.get_forecast(steps=steps)
            pred_mean = pred.predicted_mean
            try:
                conf_int = pred.conf_int()
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]
            except Exception:
                resid_std = np.std(self.model_fit.resid.dropna())
                lower = pred_mean - 1.96 * resid_std
                upper = pred_mean + 1.96 * resid_std
            pred_mean.index = full_index
            lower.index = full_index
            upper.index = full_index

        else:
            pred = self.model_fit.get_forecast(steps=steps, exog=exog_forecast)
            pred_mean = pred.predicted_mean
            try:
                conf_int = pred.conf_int()
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]
            except Exception:
                resid_std = np.std(self.model_fit.resid.dropna())
                lower = pred_mean - 1.96 * resid_std
                upper = pred_mean + 1.96 * resid_std
            pred_mean.index = full_index
            lower.index = full_index
            upper.index = full_index

        df_forecast = pd.DataFrame({
            '預測值': pred_mean,
            '下限': lower,
            '上限': upper
        }, index=full_index)

        return df_forecast

    def calculate_metrics(self, actual, predicted):
        residuals = actual - predicted
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs(residuals / actual.replace(0, np.nan))) * 100
            smape = 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
        max_ae = np.max(np.abs(residuals))
        r2 = r2_score(actual, predicted)
        n = len(actual)
        p = len(self.model_fit.params) if hasattr(self.model_fit, 'params') else 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
        stabilized_r2 = 1 - (np.var(residuals) / np.var(actual)) if np.var(actual) != 0 else np.nan

        bic = getattr(self.model_fit, 'bic', np.nan)
        aic = getattr(self.model_fit, 'aic', np.nan)
        normalized_bic = bic / n if not np.isnan(bic) else np.nan

        try:
            lb_test = acorr_ljungbox(self.model_fit.resid.dropna(), lags=[10], return_df=True)
            ljung_box_p = lb_test['lb_pvalue'].values[0]
        except Exception:
            ljung_box_p = np.nan

        try:
            dw_stat = durbin_watson(self.model_fit.resid.dropna())
        except Exception:
            dw_stat = np.nan

        metrics = {
            '樣本數 N': n,
            'R-squared': r2,
            'Adjusted R-squared': adj_r2,
            'Stabilized R-squared': stabilized_r2,
            'MAPE (%)': mape,
            'SMAPE (%)': smape,
            'RMSE': rmse,
            'MAE': mae,
            'Max AE': max_ae,
            'Normalized BIC': normalized_bic,
            'AIC': aic,
            'Ljung-Box p-value': ljung_box_p,
            'Durbin-Watson': dw_stat
        }
        return metrics

    def plot_acf_pacf(self, train_start, train_end):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        train_df = self.df.loc[train_start:train_end]
        series = train_df[self.target_col].dropna()
        plot_acf(series, ax=axes[0, 0], lags=30)
        axes[0, 0].set_title('原始資料 ACF')
        plot_pacf(series, ax=axes[0, 1], lags=30)
        axes[0, 1].set_title('原始資料 PACF')
        if self.model_fit is not None:
            try:
                resid = self.model_fit.resid.dropna()
                plot_acf(resid, ax=axes[1, 0], lags=30)
                axes[1, 0].set_title('模型殘差 ACF')
                plot_pacf(resid, ax=axes[1, 1], lags=30)
                axes[1, 1].set_title('模型殘差 PACF')
            except Exception:
                axes[1, 0].set_visible(False)
                axes[1, 1].set_visible(False)
        plt.tight_layout()
        return fig

    @staticmethod
    def summarize_quality(metrics_dict):
        summary = []
        if metrics_dict.get('R-squared', 0) < 0.5:
            summary.append("R平方過低，建議增加訓練資料量或加入其他外生變數。")
        if metrics_dict.get('MAPE (%)', 100) > 20:
            summary.append("MAPE過高，建議考慮目標變數轉換或加入更多外部影響因素。")
        if metrics_dict.get('Ljung-Box p-value', 0) <= 0.05:
            summary.append("殘差自相關明顯，建議提高模型的AR/季節性階數或增加外生變數。")

        if not summary:
            return "✅ 模型表現良好，可直接應用於實務預測。"
        else:
            return "❌ 模型存在以下問題：\n" + "\n".join(f"- {s}" for s in summary)
