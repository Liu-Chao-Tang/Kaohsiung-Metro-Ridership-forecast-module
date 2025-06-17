import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(filepath):
    df = pd.read_excel(filepath)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    return df

def train_model(df, target_col='總運量', n_days=7, order=(1,1,1), seasonal_order=(0,0,0,0), use_exog=[], exog_future=None):
    df = df.copy()
    df = df.dropna(subset=[target_col])
    train = df[:-n_days]
    test = df[-n_days:]
    
    # 外生變數改成你的新欄位名稱
    valid_exog = ['變數1-特殊節日', '變數2-大型活動', '變數3-颱風假']
    selected_exog = [col for col in use_exog if col in valid_exog]
    
    exog_train = train[selected_exog] if selected_exog else None
    exog_test = test[selected_exog] if selected_exog else None

    model = SARIMAX(train[target_col], order=order, seasonal_order=seasonal_order, exog=exog_train)
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=n_days, exog=exog_test)
    pred = forecast.predicted_mean
    conf_int = forecast.conf_int()

    actual = test[target_col].reset_index(drop=True)

    return {
        'forecast': pred,
        'conf_int': conf_int,
        'actual': actual,
        'model': result
    }

def calculate_metrics(actual, forecast):
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    max_ae = np.max(np.abs(actual - forecast))
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Max AE': max_ae
    }
