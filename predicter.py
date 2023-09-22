import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype


def intrinsic_value(df):
    interest_rate = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
                                )
    interest_rate = interest_rate.set_axis(['Date', 'Discount_rate'], axis=1)
    interest_rate['Date'] = pd.to_datetime(interest_rate['Date'])
    interest_rate['Discount_rate'] = interest_rate['Discount_rate']/100
    interest_rate['Date'] = interest_rate['Date'].dt.to_period('M').dt.end_time
    interest_rate['Date'] = interest_rate['Date'].dt.strftime('%Y-%m-%d')
    interest_rate['Date'] = pd.to_datetime(interest_rate['Date'])
    df = df.merge(interest_rate, left_on='Date',
                  right_on='Date', how='left')
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 0 if x == '' else x)
    for col in df.columns:
        if is_object_dtype(df[col]):
            df[col] = df[col].astype(float)
    for row in range(len(df)-1, -1, -1):
        if row == df.shape[0]-1:
            df.loc[row, 'Discount_factor'] = 1
        else:
            df.loc[row, 'Discount_factor'] = 1 / ((1 + df.loc[row, 'Discount_rate']/4) ** ((len(df) - row)))
    df['DCF'] = df['FCF']*df['Discount_factor']
    for row in range(len(df)-1, -1, -1):
        if row == df.shape[0]-1:
            df.loc[row, 'Grow_rate'] = 1
            df.loc[row, 'CDCF'] = df.loc[row, 'DCF']
        else:
            df.loc[row, 'Grow_rate'] = (df.loc[row, 'FCF']/df.loc[row+1, 'FCF']-1)
            df.loc[row, 'CDCF'] = df.loc[row+1, 'CDCF']+df.loc[row, 'DCF']
    df['Market_value'] = (df['FCF']/(df['Discount_rate']/4-df['Grow_rate']))*df['Discount_factor']+df['CDCF']
    df['Intrinsic Value'] = df['Market_value']/df['Shares Outstanding']
    # q1 = np.percentile(df['Intrinsic Value'], 25)
    # q3 = np.percentile(df['Intrinsic Value'], 75)
    # iqr = q3 - q1
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr
    # df.loc[(df['Intrinsic Value'] > upper_bound) | (df['Intrinsic Value'] < lower_bound), 'Intrinsic Value'] = np.nan
    # df['Intrinsic Value'].interpolate(method='nearest', inplace=True)
    return df
