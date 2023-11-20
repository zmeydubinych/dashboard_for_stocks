from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


def intrinsic_value_curr(df: pd.DataFrame, data_stock):
    """ Preproccessing data after scrapping
    :type arg1: DataFrame.
    :type arg2: DataFrame.
    :return: Return preprocceed DataFrame.
    :rtype: DataFrame.
    """

    # replace missing value with 0
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if x == '' else x)

    # transform object data type to float
    for col in df.columns:
        if is_object_dtype(df[col]):
            df[col] = df[col].astype(float)

    # calculate discount factor, discounted cash flow (DCF), grow_rate, cammulative disounted cash flow (CDCF)
    for row in range(len(df)-1, -1, -1):
        if row == df.shape[0]-1:
            df.loc[row, 'Discount_factor'] = 1
            df.loc[row, 'Grow_rate'] = 1
            df.loc[row, 'DCF'] = df.loc[row, 'FCF'] * \
                df.loc[row, 'Discount_factor']  # type: ignore
            df.loc[row, 'CDCF'] = df.loc[row, 'DCF']
        else:
            df.loc[row, 'Discount_factor'] = 1 / \
                (((1 + df.loc[row, 'Discount_rate']) ** 0.25)  # type: ignore
                 ** (len(df) - row))  # type: ignore
            if df.loc[row, 'Discount_factor'] > df.loc[row+1, 'Discount_factor']:  # type: ignore
                df.loc[row, 'Discount_factor'] = df.loc[row +
                                                        1, 'Discount_factor']  # type: ignore
            df.loc[row, 'Grow_rate'] = (
                df.loc[row, 'FCF'] - df.loc[row+1, 'FCF'])/(abs(df.loc[row+1, 'FCF'])+1e-10)  # type: ignore
            df.loc[row, 'DCF'] = df.loc[row, 'FCF'] * \
                df.loc[row, 'Discount_factor']  # type: ignore
            df.loc[row, 'CDCF'] = df.loc[row+1, 'CDCF'] + \
                df.loc[row, 'DCF']  # type: ignore

    # calculate capitalization rate via stock price and operating income
    for row in range(len(df)-1, -1, -1):
        start = df.loc[row, 'Date']
        price_last_quarter = data_stock.loc[data_stock["Date"]
                                            == start+timedelta(days=1), 'Close'].item()  # type: ignore
        df.loc[row, 'Cap_rate'] = df['Operating Income'].astype(
            float)[row]/(price_last_quarter * df['Shares Outstanding'].astype(float)[row]+1e-10)
        if df.loc[row, 'Cap_rate'] < 0.01:  # type: ignore
            df.loc[row, 'Cap_rate'] = 0.01

    # calculate ROA, ROA_market, FR_market
    df['ROA'] = df['Net Income/Loss']/df['Total Assets']

    for row in range(len(df)-1, -1, -1):
        if row == df.shape[0]-1:
            df.loc[row, 'ROA_market'] = 1e-10
            df.loc[row, 'FR_market'] = 1e-10
        else:
            df.loc[row, 'ROA_market'] = (df.loc[row, 'SP500'] - df.loc[row+1, 'SP500'])/(abs(df.loc[row+1, 'SP500'])+1e-10)  # type:ignore
            df.loc[row, 'FR_market'] = df.loc[row:len(
                df)-1, 'ROA_market'].median()

    # calculate Beta_return
    for row in range(len(df)-1, -1, -1):
        if row >= df.shape[0]-10:
            covariance = np.cov(df.loc[df.shape[0]-10:len(df)-1, 'ROA'],
                                df.loc[df.shape[0]-10:len(df)-1, 'ROA_market'])[0][1]
            variance = np.var(df.loc[df.shape[0]-10:len(df)-1, 'ROA_market'])
            df.loc[row, 'Beta_return'] = covariance / variance
        else:
            covariance = np.cov(
                df.loc[row:row+10, 'ROA'], df.loc[row:row+10, 'ROA_market'])[0][1]
            variance = np.var(df.loc[row:row+10, 'ROA_market'])
            df.loc[row, 'Beta_return'] = covariance / variance

    df['FR_rate'] = df['Discount_rate']+df['Beta_return'] * \
        (df['FR_market']-df['Discount_rate'])

    # Market value, intrinsic value
    df['Market_value'] = (df['FCF']/df['Cap_rate']) * \
        df['Discount_factor']+df['CDCF']  # market value via cap_rate

    # df['Market_value'] = (df['FCF']/df['FR_rate']) * \
    #     df['Discount_factor']+df['CDCF']  # market value via future_return_rate for future use only

    df['Intrinsic Value'] = df['Market_value']/df['Shares Outstanding']

    return df


def create_sequences(data, seq_length):
    """Create sequence from Data"""
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def next_quarter_date(date):
    """From datetime find out last day of next quarter"""
    quarter_month = date.month+3
    next_quarter = datetime(date.year + quarter_month //
                            12, quarter_month % 12 + 1, 1)
    return (next_quarter - timedelta(days=1))


def intrinsic_value_next(df: pd.DataFrame, data_stock):
    """Prediction for next quarter"""
    df = intrinsic_value_curr(df, data_stock)
    data = df['FCF'][::-1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(
        data.values.reshape(-1, 1))  # type: ignore
    seq_length = 4
    X_train, y_train = create_sequences(scaled_data, seq_length)
    X_test = scaled_data[-4-seq_length:-4]
    X_test = X_test.reshape(1, seq_length, -1)
    model = Sequential([
        LSTM(4, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(4),
        Dense(1)
    ])
    # maybe mae should use instead of mse? cause of outliers
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=350, verbose=False)  # type: ignore

    next_fcf = np.squeeze(scaler.inverse_transform(
        model.predict(X_test).reshape(-1, 1)))
    next_disc_rate = df['Discount_rate'].loc[0]
    next_disc_factor = 1 / (((1 + next_disc_rate) ** 0.25) ** len(df))
    next_dcf = next_fcf*next_disc_factor
    next_cdcf = df.loc[0, 'CDCF']+next_dcf

    price_last_quarter = data_stock.loc[0, 'Close']  # type:ignore
    cap_rate = df['Operating Income'].astype(
        float)[0]/(price_last_quarter * df['Shares Outstanding'].astype(float)[0])
    cap_rate = max(cap_rate, 0.01)

    next_market_value = next_fcf/cap_rate*next_disc_factor+next_cdcf
    next_intrinsic_value = next_market_value/df['Shares Outstanding'].loc[0]

    prediction_df = pd.DataFrame.from_dict({'Date': next_quarter_date(df.loc[0, 'Date']),
                                            'FCF': next_fcf, 'Discount_rate': next_disc_rate,
                                            'Discount factor': next_disc_factor,
                                            'DCF': next_dcf, 'CDCF': next_cdcf,
                                            'Market value': next_market_value,
                                            'Intrinsic value': next_intrinsic_value.item()}, orient='index'
                                             ).transpose()
    return prediction_df
