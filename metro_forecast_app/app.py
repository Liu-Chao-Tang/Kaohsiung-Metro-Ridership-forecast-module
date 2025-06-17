import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(filepath):
    df = pd.read_excel(filepath)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    return df

def train_model(df, target_col='總運量', n_days=7, order=(1,1,1), seasonal_order=(0,0,0,0), use_exog=[]):
    df = df.copy()
    df = df.dropna(subset=[target_col])
    train = df[:-n_days]
    test = df[-n_days:]
    
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

    return pred, conf_int, actual, result

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

def main():
    st.title("捷運運量預測系統")

    df = load_data("data/daily_volume.xlsx")
    st.subheader("資料預覽")
    st.dataframe(df.head())

    st.sidebar.header("模型設定")
    
    # 外生變數勾選
    exog_options = ['變數1-特殊節日', '變數2-大型活動', '變數3-颱風假']
    use_exog = st.sidebar.multiselect("選擇外生變數", exog_options)

    # 預測天數輸入
    n_days = st.sidebar.number_input("預測天數 (1~30)", min_value=1, max_value=30, value=7)

    # 按鈕觸發預測
    if st.sidebar.button("開始預測"):
        with st.spinner("模型訓練中..."):
            pred, conf_int, actual, model_result = train_model(df, n_days=n_days, use_exog=use_exog)
            metrics = calculate_metrics(actual, pred)

        st.subheader("預測結果")
        st.line_chart(pd.DataFrame({
            '實際值': actual,
            '預測值': pred
        }))

        st.subheader("95%信賴區間")
        st.dataframe(conf_int)

        st.subheader("模型績效指標")
        st.write(metrics)

if __name__ == "__main__":
    main()
