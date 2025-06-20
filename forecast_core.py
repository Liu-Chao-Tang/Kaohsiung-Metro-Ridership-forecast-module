# forecast_core.py

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ForecastModel:
    def __init__(self, df, target_col, use_exog=False, exog_cols=None):
        """
        初始化 ForecastModel 物件

        Args:
            df (pd.DataFrame): 時間序列資料，索引為日期
            target_col (str): 目標預測欄位名稱
            use_exog (bool): 是否使用外生變數
            exog_cols (list): 外生變數欄位名稱清單
        """
        self.df = df
        self.target_col = target_col
        self.use_exog = use_exog
        self.exog_cols = exog_cols if exog_cols else []
        self.model_fit = None
        self.order = None
        self.seasonal_order = None

    def load_data(self, filepath):
        """載入資料"""
        self.df = pd.read_excel(filepath, index_col=0, parse_dates=True)

    def auto_fit(self, train_start, train_end, m=7):
        """
        使用 auto_arima 自動搜尋最佳模型參數

        Args:
            train_start (datetime): 訓練開始日期
            train_end (datetime): 訓練結束日期
            m (int): 季節性週期 (預設7天)

        Returns:
            order (tuple): (p,d,q)
            seasonal_order (tuple): (P,D,Q,s)
        """
        train_df = self.df.loc[train_start:train_end]
        endog = train_df[self.target_col]
        exog = train_df[self.exog_cols] if self.use_exog and self.exog_cols else None

        stepwise_model = auto_arima(
            endog, exogenous=exog, seasonal=True, m=m,
            start_p=0, max_p=3, start_q=0, max_q=3,
            start_P=0, max_P=2, start_Q=0, max_Q=2,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
        self.order = stepwise_model.order
        self.seasonal_order = stepwise_model.seasonal_order
        return self.order, self.seasonal_order

    def fit(self, train_start, train_end, order, seasonal_order, model_type="SARIMAX"):
        """
        訓練模型

        Args:
            train_start (datetime): 訓練開始日期
            train_end (datetime): 訓練結束日期
            order (tuple): (p,d,q)
            seasonal_order (tuple): (P,D,Q,s)
            model_type (str): 模型類型 ('AR', 'MA', 'ARIMA', 'SARIMAX')

        Returns:
            model_fit: 訓練好的模型物件
        """
        train_df = self.df.loc[train_start:train_end]
        endog = train_df[self.target_col]
        exog = train_df[self.exog_cols] if self.use_exog and self.exog_cols else None

        self.order = order
        self.seasonal_order = seasonal_order

        if model_type == "AR":
            model = AutoReg(endog, lags=order[0], old_names=False)
            self.model_fit = model.fit()
        elif model_type == "MA":
            model = ARIMA(endog, order=(0, 0, order[2]))
            self.model_fit = model.fit()
        elif model_type == "ARIMA":
            model = ARIMA(endog, order=order)
            self.model_fit = model.fit()
        else:  # SARIMAX
            model = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                            exog=exog, enforce_stationarity=False, enforce_invertibility=False)
            self.model_fit = model.fit(disp=False)
        return self.model_fit

    def forecast(self, forecast_start, forecast_end):
        """
        進行未來期間預測，並產生動態信賴區間

        Args:
            forecast_start (datetime): 預測起始日期
            forecast_end (datetime): 預測結束日期

        Returns:
            pd.DataFrame: 包含預測值、下限、上限的資料表，index為日期
        """
        if self.model_fit is None:
            raise ValueError("模型尚未訓練完成")

        full_index = pd.date_range(forecast_start, forecast_end)
        valid_index = full_index.intersection(self.df.index)

        if isinstance(self.model_fit, (AutoReg, ARIMA)):
            pred = self.model_fit.get_prediction(start=valid_index[0], end=valid_index[-1])
        else:
            exog_forecast = self.df.loc[valid_index, self.exog_cols] if self.use_exog and self.exog_cols else None
            pred = self.model_fit.get_forecast(steps=len(valid_index), exog=exog_forecast)

        pred_mean = pred.predicted_mean

        # 嘗試取得模型信賴區間，若無則用殘差標準差動態估計
        try:
            conf_int = pred.conf_int()
        except Exception:
            resid_std = np.std(self.model_fit.resid.dropna()) if hasattr(self.model_fit, 'resid') else np.std(pred_mean) * 0.1
            margin = 1.96 * resid_std
            lower = pred_mean - margin
            upper = pred_mean + margin
            conf_int = pd.DataFrame({'lower': lower, 'upper': upper}, index=valid_index)

        df_forecast = pd.DataFrame({
            '預測值': pred_mean,
            '下限': conf_int.iloc[:, 0],
            '上限': conf_int.iloc[:, 1]
        }, index=valid_index)

        return df_forecast

    def calculate_metrics(self, actual, predicted):
        """
        計算模型績效指標

        Args:
            actual (pd.Series): 實際值
            predicted (pd.Series): 預測值

        Returns:
            dict: 指標字典
        """
        residuals = actual - predicted
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs(residuals / actual.replace(0, np.nan))) * 100
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

        metrics = {
            'R-squared': r2,
            'Adjusted R-squared': adj_r2,
            'Stabilized R-squared': stabilized_r2,
            'MAPE (%)': mape,
            'RMSE': rmse,
            'MAE': mae,
            'Max AE': max_ae,
            'Normalized BIC': normalized_bic,
            'AIC': aic,
            'Ljung-Box p-value': ljung_box_p
        }
        return metrics

    def plot_acf_pacf(self, train_start, train_end):
        """
        繪製訓練期間資料及殘差的 ACF / PACF 圖

        Args:
            train_start (datetime): 訓練起始日
            train_end (datetime): 訓練結束日

        Returns:
            matplotlib.figure.Figure: 繪圖物件
        """
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
        """
        根據指標簡單給出模型品質摘要

        Args:
            metrics_dict (dict): 模型績效指標字典

        Returns:
            str: 模型品質摘要訊息
        """
        summary = []
        if metrics_dict.get('R-squared', 0) < 0.5:
            summary.append("R平方過低，建議增加訓練資料量或加入其他外生變數。")
        if metrics_dict.get('MAPE (%)', 100) > 20:
            summary.append("MAPE過高，建議考慮目標變數轉換或加入更多外部影響因素。")
        if metrics_dict.get('Ljung-Box p-value', 0) <= 0.05:
            summary.append("殘差自相關明顯，建議提高模型的AR或季節性階數。")

        if not summary:
            return "✅ 模型表現良好，可直接應用於實務預測。"
        else:
            return "❌ 模型存在以下問題：\n" + "\n".join(f"- {s}" for s in summary)
