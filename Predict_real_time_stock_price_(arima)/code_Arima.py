import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

def yfinance_data(company_id):

    tickers = yf.Ticker(company_id)

    series_data_historical = tickers.history(period="max")
    df_1 = pd.DataFrame(series_data_historical)

    df_1["company_id"] = company_id
    df_1["series_type"] = "historical_data"

    df_1 = df_1.sort_index(ascending=True)
    # df_1 = df_1.loc[:'20190101']
    historical_price = df_1["Close"]

    return historical_price


def model_arima(series):

    forecast_periods = 365
    train_data = series.iloc[0:-forecast_periods]
    test_data = series.iloc[-forecast_periods:]

    model_x = ARIMA(train_data, order=(2, 2, 1))
    model_x = model_x.fit()  # disp=0 # Fit the model set previously

    results_x = model_x.get_forecast(steps=forecast_periods)  # PredictionResultsWrapper
    forecast, stderr, conf = results_x.predicted_mean, results_x.var_pred_mean, results_x.conf_int()

    forecast.index = test_data.index
    conf.index = conf.index

    fig, ax = plt.subplots()
    ax.plot(train_data, label='Train Series', color='b')  # test data to compare
    ax.plot(test_data, label='Test Series', color='k')  # test data to compare
    ax.plot(forecast, label='prediction', color='r')  # # forecast value

    x = forecast.index
    ax.fill_between(x, conf.loc[:, 'lower Close']  # x valeus ; y1 - line 1
                    , conf.loc[:, 'upper Close']
                    # #, y2 - line 2 # or np.mean(y) # to add overall median to distinct both
                    , interpolate=False  # , interpolate=
                    # , where=(y > np.mean(y))
                    , color='b', alpha=.25
                    , label='95%% prediction Interval')

    ax.yaxis.grid(True, linestyle='--', color='c', alpha=0.4)  # horizontal lines
    plt.title(f"{company_id}: Future price prediction")
    plt.xlabel('Date')
    plt.ylabel('Actual Series')
    plt.legend(loc='upper left', fontsize=8)
    plt.gcf().autofmt_xdate()

    plt.tight_layout()

    plt.savefig("plot2.png")
    plt.show()


company_id = "GOOGL"
series = yfinance_data(company_id)

model_arima(series) 
