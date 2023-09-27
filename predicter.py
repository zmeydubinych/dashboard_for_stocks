import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


def intrinsic_value_curr(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if x == '' else x)
    for col in df.columns:
        if is_object_dtype(df[col]):
            df[col] = df[col].astype(float)
    for row in range(len(df)-1, -1, -1):
        if row == df.shape[0]-1:
            df.loc[row, 'Discount_factor'] = 1
        else:
            df.loc[row, 'Discount_factor'] = 1 / \
                ((1 + df.loc[row, 'Discount_rate']/4) ** ((len(df) - row)))
    df['DCF'] = df['FCF']*df['Discount_factor']
    for row in range(len(df)-1, -1, -1):
        if row == df.shape[0]-1:
            df.loc[row, 'Grow_rate'] = 1
            df.loc[row, 'CDCF'] = df.loc[row, 'DCF']
        else:
            df.loc[row, 'Grow_rate'] = (
                df.loc[row, 'FCF']/df.loc[row+1, 'FCF']-1)
            df.loc[row, 'CDCF'] = df.loc[row+1, 'CDCF']+df.loc[row, 'DCF']
    df['Market_value'] = (df['FCF']/(df['Discount_rate'] /
                          4-df['Grow_rate']))*df['Discount_factor']+df['CDCF']
    df['Intrinsic Value'] = df['Market_value']/df['Shares Outstanding']
    q1 = np.percentile(df['Intrinsic Value'], 25)
    q3 = np.percentile(df['Intrinsic Value'], 75)
    iqr = q3 - q1
    lower_bound = q1 - 2 * iqr
    upper_bound = q3 + 2 * iqr
    df.loc[(df['Intrinsic Value'] > upper_bound) | (
        df['Intrinsic Value'] < lower_bound), 'Intrinsic Value'] = np.nan
    df['Intrinsic Value'].interpolate(method='nearest', inplace=True)
    return df


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def intrinsic_value_next(df):
    data = df['FCF'][::-1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    seq_length = 4
    X_train, y_train = create_sequences(scaled_data, seq_length)
    X_test = scaled_data[-4-seq_length:-4]
    X_test = X_test.reshape(1, seq_length, -1)
    model = Sequential([
        LSTM(4, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(4),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history=model.fit(X_train, y_train, epochs=350, verbose=False)
    next_fcf = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1))
    next_disc_rate = df['Discount_rate'].loc[0]
    next_disc_factor = 1 / ((1 + next_disc_rate/4) ** ((len(df))))
    next_dcf = next_fcf*next_disc_factor
    next_cdcf = df.loc[0, 'CDCF']+next_dcf
    next_market_value = (
        next_fcf/(next_disc_rate/4-df['Grow_rate'].loc[0]))*next_disc_factor+next_cdcf
    next_intrinsic_value = next_market_value/df['Shares Outstanding'].loc[0]
    result=[next_intrinsic_value.item(), history.history["loss"][-1]]
    return result
